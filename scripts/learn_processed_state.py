import time
from pathlib import Path
from typing import override

import h5py  # type: ignore[import-untyped]
import matplotlib.pyplot as plt
import numpy as np
import torch
from multiscat import OptimizationConfig
from multiscat.basis import (
    close_coupling_basis,
    scattering_metadata_from_stacked_delta_x,
    split_scattering_metadata,
)
from multiscat.config import (
    MorseScatteringCondition,
    UnitSystem,
    condition_in_natural_units,
    get_condition_natural_units,
    incident_k_from_angles,
)
from multiscat.multiscat import (
    get_full_b_wave,
    get_preconditioned_state_from_state,
    get_scattering_state,
)
from scipy.constants import (  # type: ignore[import-untyped]
    angstrom,
    electron_volt,
    physical_constants,
)
from scipy.constants import angstrom as angstrom_si  # type: ignore[import-untyped]
from slate_core import (
    Array,
    basis,
    plot,
)
from slate_core.basis import AsUpcast
from slate_core.metadata import (
    Domain,
    LobattoSpacedLengthMetadata,
)
from slate_core.metadata.volume import fundamental_stacked_delta_x
from slate_quantum import operator
from torch import nn, optim
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from tqdm import tqdm

from multiscat_ml import plot_loss_curves
from multiscat_ml.model_zoo import ModelZooEntry, compare_model_validation_loss
from multiscat_ml.utils import (
    TrainingStats,
)

# Constants
HELIUM_MASS = physical_constants["alpha particle mass"][0]
HELIUM_ENERGY = 7 * electron_volt * 10**-3
Z_HEIGHT = 8


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")  # pyright: ignore[reportConstantRedefinition]
else:
    DEVICE = torch.device("cpu")  # pyright: ignore[reportConstantRedefinition]

PARAMS_MIN = np.array(
    [0.1],
    dtype=np.float64,
)
PARAMS_MAX = np.array(
    [2.0],
    dtype=np.float64,
)

HELIUM_MASS = physical_constants["alpha particle mass"][0]
HELIUM_ENERGY = 20 * electron_volt * 10**-3

UNIT_CELL = 2.84 * angstrom
Z_HEIGHT = 8 * angstrom


def denormalize_params(
    params_norm: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    """Scales parameters back to their original physical units."""
    return params_norm * (PARAMS_MAX - PARAMS_MIN) + PARAMS_MIN


def get_stable_state(
    condition: MorseScatteringCondition,
    config: OptimizationConfig,
) -> np.ndarray[tuple[int, int, int], np.dtype[np.complex128]]:
    """Simulate the preconditioned state from the given ScatteringCondition."""
    converted_condition = condition_in_natural_units(condition)
    b = get_full_b_wave(
        converted_condition.metadata,
        converted_condition.incident_k,
    )

    # Get the scattering state
    state = get_scattering_state(condition, config)

    # Get the preconditioned state
    preconditioned_state = get_preconditioned_state_from_state(
        state,
        condition,
        n_channels=config.n_channels,
    )

    # Return the real and imaginary parts of the preconditioned state
    preconditioned_data = preconditioned_state.with_basis(
        close_coupling_basis(condition.metadata),
    ).raw_data.reshape(condition.metadata.shape)
    return 2.0j * b * preconditioned_data


def condition_from_params(
    params: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> MorseScatteringCondition:
    """Convert a tensor of parameters into a ScatteringCondition."""
    (energy_factor,) = denormalize_params(params)

    morse_parameters = operator.build.CorrugatedMorseParameters(
        depth=7.63 * electron_volt * 10**-3,
        height=0.91 * angstrom,
        offset=3.0 * angstrom,
        beta=0.10,
    )

    metadata = scattering_metadata_from_stacked_delta_x(
        (
            np.array([UNIT_CELL / np.sqrt(2), 0, 0]),
            np.array([UNIT_CELL / np.sqrt(8), np.sqrt(3) * UNIT_CELL / np.sqrt(8), 0]),
            np.array([0, 0, Z_HEIGHT]),
        ),
        (18, 18, 200),
    )
    condition = MorseScatteringCondition(
        mass=3 * HELIUM_MASS,
        incident_k=incident_k_from_angles(
            mass=3 * HELIUM_MASS,
            energy=HELIUM_ENERGY * energy_factor,
            theta=np.deg2rad(30),
            phi=np.deg2rad(60),
        ),
        metadata=metadata,
        morse_parameters=morse_parameters,
    )
    return condition.with_units(get_condition_natural_units(condition))


def normalize_params(
    params: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    """Scales parameters to a [0, 1] range."""
    return (params - PARAMS_MIN) / (PARAMS_MAX - PARAMS_MIN)


def params_from_condition(
    condition: MorseScatteringCondition,
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    """Extract the parameters from a ScatteringCondition."""
    condition = condition.with_units(UnitSystem())

    energy = condition.incident_energy / HELIUM_ENERGY
    return normalize_params(np.array([energy]))


def simulate_stable_state(
    params: np.ndarray[tuple[int], np.dtype[np.float64]],
    *,
    n_samples_per_slice: int = 100,
) -> tuple[
    np.ndarray[tuple[int], np.dtype[np.complex128]],
    np.ndarray[tuple[int, int], np.dtype[np.floating]],
]:
    """Simulate the stable state from the given parameters."""
    condition = condition_from_params(params)
    config = OptimizationConfig(precision=1e-5, max_iterations=1000, n_channels=160)
    state_k_space = get_stable_state(condition, config)

    metadata_x01, metadata_z = split_scattering_metadata(condition.metadata)

    delta_x0, delta_x1 = fundamental_stacked_delta_x(metadata_x01)

    n_kx, n_ky, n_z = state_k_space.shape

    # 1. Sample fractional coordinates u, v in [0, 1) for all z slices in one step
    rng = np.random.default_rng()
    u = rng.uniform(0.0, 1.0, size=(n_z, n_samples_per_slice))
    v = rng.uniform(0.0, 1.0, size=(n_z, n_samples_per_slice))

    # 2. Compute physical (x, y, z) spatial positions
    x = u * delta_x0[0] + v * delta_x1[0]
    y = u * delta_x0[1] + v * delta_x1[1]
    z = np.broadcast_to(metadata_z.values[:, None], (n_z, n_samples_per_slice))

    coords_out = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=0)

    # 3. Compute reciprocal lattice vectors B = 2*pi * (A^-1)^T
    dk_stacked = 2 * np.pi * np.linalg.inv(np.column_stack([delta_x0, delta_x1])).T

    # 4. Construct wavevector grid (kx, ky) in reciprocal space
    m_freq = np.fft.fftfreq(n_kx) * n_kx
    n_freq = np.fft.fftfreq(n_ky) * n_ky

    kx = m_freq[:, None] * dk_stacked[0, 0] + n_freq[None, :] * dk_stacked[0, 1]
    ky = m_freq[:, None] * dk_stacked[1, 0] + n_freq[None, :] * dk_stacked[1, 1]

    # 5. Calculate (k dot x) explicitly via broadcasting: shape (n_kx, n_ky, n_z, 100)
    k_dot_x = (
        kx[:, :, None, None] * x[None, None, :, :]
        + ky[:, :, None, None] * y[None, None, :, :]
    )

    # 6. Evaluate sum_{k} state(k) * exp(i * k dot x) normalized by grid size
    state_4d = state_k_space[:, :, :, None]
    sampled_state = np.sum(state_4d * np.exp(1j * k_dot_x), axis=(0, 1))

    state_out = sampled_state.ravel()

    return state_out, coords_out


def generate_dataset_hdf5(
    filepath: Path, num_samples: int = 500, n_samples_per_slice: int = 100
) -> None:
    """Generate parameters and Preconditioned state, saving them directly to disk."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if filepath.exists():
        print(f"Dataset already exists at {filepath}. Skipping generation.")
        return

    print(
        f"Generating {num_samples} samples straight to disk. This may take a while...",
    )
    rng = np.random.default_rng()

    with h5py.File(filepath, "w") as f:
        # Generate the first sample to inspect output dimensions dynamically
        # Note: we currently assume a fixed nz = 200
        n_points = 200 * n_samples_per_slice

        # Inputs (x, y, z, energy_factor) for each sampled condition and coordinate
        x_ds = f.create_dataset("X", shape=(num_samples, n_points, 4), dtype=np.float64)

        # Outputs (real, imag) of the sampled state at each coordinate
        y_ds = f.create_dataset("Y", shape=(num_samples, n_points, 2), dtype=np.float32)

        for i in range(num_samples):
            print(f"Generating sample {i + 1}/{num_samples}")

            params = rng.uniform(size=1)
            state, coords = simulate_stable_state(
                params, n_samples_per_slice=n_samples_per_slice
            )

            energies = np.full((n_points, 1), params[0])
            x_ds[i] = np.hstack([coords.T, energies])

            # Store real and imaginary parts as (N, 2) in float32
            y_ds[i] = np.column_stack([state.real, state.imag]).astype(np.float32)


class HDF5ScatteringDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """PyTorch Dataset wrapper for reading (X, Y) pairs from a scattering HDF5 file."""

    def __init__(self, filepath: Path) -> None:
        self.filepath = filepath
        self._file = None

        # Inspect length during initialization
        with h5py.File(self.filepath, "r") as f:
            self._length = len(f["X"])

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:  # ty: ignore[invalid-method-override]
        # Lazy file loading avoids HDF5 file-handle serialization issues with DataLoader workers
        if self._file is None:
            self._file = h5py.File(self.filepath, "r")

        x_arr = self._file["X"][idx]
        y_arr = self._file["Y"][idx]

        # Convert to float32 tensors for PyTorch model compatibility
        x_tensor = torch.from_numpy(x_arr).to(torch.float32)
        y_tensor = torch.from_numpy(y_arr).to(torch.float32)

        return x_tensor, y_tensor

    def __del__(self) -> None:
        if self._file is not None:
            self._file.close()


def generate() -> None:
    for i in range(50):
        data_path = Path(f"data/15/stable_state_data_{i}.hdf5")
        generate_dataset_hdf5(data_path, num_samples=500)


def load_datasets() -> ConcatDataset[tuple[torch.Tensor, torch.Tensor]]:
    datasets = [
        HDF5ScatteringDataset(Path(f"data/15/stable_state_data_{i}.hdf5"))
        for i in range(50)
    ]
    return ConcatDataset[tuple[torch.Tensor, torch.Tensor]](datasets)


class SirenLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        is_first: bool = False,  # ruff: ignore[boolean-default-value-positional-argument, boolean-type-hint-positional-argument]
        omega_0: float = 5.0,
    ) -> None:
        super().__init__()

        self.linear = nn.Linear(in_features, out_features)
        self.omega_0 = omega_0
        self.is_first = is_first

        self.init_weights()

    def init_weights(self) -> None:
        with torch.no_grad():
            if self.is_first:
                bound = 1 / self.linear.in_features
            else:
                bound = np.sqrt(6 / self.linear.in_features) / self.omega_0

            self.linear.weight.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


class PureSIREN(nn.Module):
    def __init__(  # ruff: ignore[too-many-arguments, too-many-positional-arguments]
        self,
        in_dim: int = 6,
        param_dim: int | None = None,
        coord_dim: int = 0,
        hidden_dim: int = 256,
        num_layers: int = 6,
        first_omega_0: float = 5.0,
        hidden_omega_0: float = 5.0,
        output_dim: int = 1,
    ) -> None:
        super().__init__()

        if param_dim is not None:
            in_dim = param_dim + coord_dim if coord_dim > 0 else param_dim

        self.in_dim = in_dim
        self.hidden_omega_0 = hidden_omega_0

        layers = [
            SirenLayer(
                in_features=self.in_dim,
                out_features=hidden_dim,
                is_first=True,
                omega_0=first_omega_0,
            )
        ]

        layers.extend(
            SirenLayer(
                in_features=hidden_dim,
                out_features=hidden_dim,
                is_first=False,
                omega_0=hidden_omega_0,
            )
            for _ in range(num_layers - 1)
        )

        self.net = nn.ModuleList(layers)
        self.head = nn.Linear(hidden_dim, output_dim)
        self.init_head()

    def init_head(self) -> None:
        with torch.no_grad():
            bound = np.sqrt(6.0 / self.head.in_features) / self.hidden_omega_0
            self.head.weight.uniform_(-bound, bound)
            if self.head.bias is not None:
                self.head.bias.zero_()

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        for layer in self.net:
            x = layer(x)

        return self.head(x)


def _make_coords(
    Nx: int,
    Ny: int,
    Nz: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    xs = 2 * torch.arange(Nx, device=device, dtype=dtype) / Nx - 1
    ys = 2 * torch.arange(Ny, device=device, dtype=dtype) / Ny - 1
    z_domain = Domain(start=-1.0, delta=2.0)
    z_metadata = LobattoSpacedLengthMetadata(fundamental_size=Nz, domain=z_domain)
    zs_np = z_metadata.values
    zs = torch.from_numpy(zs_np).to(device=device, dtype=dtype)

    # Create the 3D grid
    grid = torch.stack(
        torch.meshgrid(xs, ys, zs, indexing="ij"),
        dim=-1,
    )  # (Nx, Ny, Nz, 3)

    return grid.reshape(-1, 3)  # (N, 3)


def sample_crop(Nx_ext, Ny_ext, Nx, Ny, device):
    sx = torch.randint(0, Nx_ext - Nx + 1, (), device=device).item()
    sy = torch.randint(0, Ny_ext - Ny + 1, (), device=device).item()
    return slice(sx, sx + Nx), slice(sy, sy + Ny)


def predict_chi_batch_from_params(
    forward_model: nn.Module,
    params_batch: torch.Tensor,
    coords: torch.Tensor,
    Nx: int,
    Ny: int,
    Nz: int,
) -> torch.Tensor:
    B = params_batch.shape[0]
    N_pts = coords.shape[0]

    coords = coords.to(device=params_batch.device, dtype=params_batch.dtype)

    # Repeat each parameter vector for every spatial point
    params_flat = (
        params_batch.unsqueeze(1)
        .expand(B, N_pts, params_batch.shape[-1])
        .reshape(B * N_pts, params_batch.shape[-1])
    )

    coords_flat = coords.unsqueeze(0).expand(B, N_pts, 3).reshape(B * N_pts, 3)

    pred_flat = forward_model(
        params=params_flat,
        coords=coords_flat,
    )

    return (
        pred_flat.view(B, N_pts, 2).transpose(1, 2).contiguous().view(B, 2, Nx, Ny, Nz)
    )


def train_model(  # ruff: ignore[too-many-locals, too-many-statements]
    model_entry: ModelZooEntry,
    epochs: int = 200,
    max_epochs_without_improvement: int = 200,
) -> None:

    model_entry.base_path.mkdir(parents=True, exist_ok=True)

    model = model_entry.model.to(DEVICE)
    forward_criterion = nn.MSELoss()
    forward_optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        forward_optimizer, mode="min", factor=0.5, patience=5
    )

    stats = TrainingStats()
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    # Load multi-file dataset directly from disk
    full_dataset = load_datasets()
    train_dataset, val_dataset = random_split(full_dataset, [0.8, 0.2])

    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
    )

    for epoch in range(epochs):
        epoch_start = time.perf_counter()
        model.train()
        train_loss = 0.0
        current_lr = forward_optimizer.param_groups[0]["lr"]

        p_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{epochs}",
            unit="batch",
        )

        for batch_idx, (batch_parameters, batch_targets) in enumerate(p_bar):
            forward_optimizer.zero_grad(set_to_none=True)

            prediction = model(batch_parameters)
            batch_loss = forward_criterion(prediction, batch_targets)
            batch_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            forward_optimizer.step()

            train_loss += batch_loss.item()
            running_loss = train_loss / (batch_idx + 1)

            p_bar.set_postfix(
                loss=f"{batch_loss.item():.3e}",
                avg=f"{running_loss:.3e}",
                lr=f"{current_lr:.1e}",
            )

        train_loss /= len(train_loader)

        # Validation step
        model.eval()
        validation_loss = 0.0
        with torch.no_grad():
            for val_params, val_targets in val_loader:
                prediction = model(val_params)
                validation_loss += forward_criterion(prediction, val_targets).item()

        validation_loss /= len(val_loader)

        scheduler.step(validation_loss)

        current_weight_decay = float(
            forward_optimizer.param_groups[0].get("weight_decay", 0.0)
        )
        stats.append(
            train_loss=train_loss,
            val_loss=validation_loss,
            weight_decay=current_weight_decay,
        )
        stats.save(model_entry.stats_path)

        epoch_time = time.perf_counter() - epoch_start
        print(
            f"Epoch {epoch + 1:03d} done | train_loss={train_loss:.6e} | "
            f"val_loss={validation_loss:.6e} | best_val={best_val_loss:.6e} | time={epoch_time:.2f}s"
        )

        if validation_loss < best_val_loss:
            best_val_loss = validation_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), model_entry.best_model_path)
            print("✓ Saved new best model checkpoint.")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= max_epochs_without_improvement:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    # Save artifacts
    stats.save(model_entry.stats_path)
    torch.save(model.state_dict(), model_entry.final_model_path)


def plot_slice_pair(
    actual_3d,
    predicted_3d,
    axis: str,
    idx: int,
    output_path: Path,
    title: str,
) -> None:
    """
    actual_3d, predicted_3d: arrays/tensors with shape (nx, ny, nz)
    axis: "xy", "xz", or "yz"
    idx: slice index along the orthogonal axis.
    """
    if hasattr(actual_3d, "detach"):
        actual_3d = actual_3d.detach().cpu().numpy()
    if hasattr(predicted_3d, "detach"):
        predicted_3d = predicted_3d.detach().cpu().numpy()

    if axis == "xy":
        actual = np.abs(actual_3d[:, :, idx])
        pred = np.abs(predicted_3d[:, :, idx])
        xlabel, ylabel = "x", "y"
    elif axis == "xz":
        actual = np.abs(actual_3d[:, idx, :])
        pred = np.abs(predicted_3d[:, idx, :])
        xlabel, ylabel = "x", "z"
    elif axis == "yz":
        actual = np.abs(actual_3d[idx, :, :])
        pred = np.abs(predicted_3d[idx, :, :])
        xlabel, ylabel = "y", "z"
    else:
        msg = f"Unknown axis: {axis}"
        raise ValueError(msg)

    error = np.abs(pred - actual)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)

    im0 = axes[0].imshow(actual.T, origin="lower", aspect="auto")
    axes[0].set_title("Actual")
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(pred.T, origin="lower", aspect="auto")
    axes[1].set_title("Predicted")
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel(ylabel)
    fig.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(error.T, origin="lower", aspect="auto")
    axes[2].set_title("Absolute error")
    axes[2].set_xlabel(xlabel)
    axes[2].set_ylabel(ylabel)
    fig.colorbar(im2, ax=axes[2])

    fig.suptitle(title, fontsize=13)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _test(
    model_name: str,
    model: nn.Module,
    output_dir: Path = Path("data/15"),
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    condition0 = MorseScatteringCondition(
        mass=HELIUM_MASS,
        morse_parameters=operator.build.CorrugatedMorseParameters(
            depth=7.63 * electron_volt * 10**-3,
            height=(1.0 / 1.1) * angstrom_si,
            offset=1.0 * angstrom_si,
            beta=0.05,
        ),
        metadata=scattering_metadata_from_stacked_delta_x(
            (
                np.array([4 * angstrom_si, 0, 0]),
                np.array([0, 4 * angstrom_si, 0]),
                np.array([0, 0, Z_HEIGHT * angstrom_si]),
            ),
            (15, 15, 200),
        ),
        incident_k=incident_k_from_angles(
            theta=np.deg2rad(0),
            phi=np.deg2rad(20),
            energy=HELIUM_ENERGY,
            mass=HELIUM_MASS,
        ),
    )

    test_params = params_from_condition(condition0)
    condition = condition_from_params(test_params)

    nx, ny, nz = condition.metadata.shape

    coords = _make_coords(
        nx,
        ny,
        nz,
        device=DEVICE,
        dtype=torch.float32,
    )

    metadata_x01, metadata_z = split_scattering_metadata(condition.metadata)
    z = metadata_z.values

    model = model.to(DEVICE)

    best_model_path = output_dir / f"{model_name}_best.pth"
    stats_path = output_dir / f"{model_name}_training_stats.pkl"
    curve_path = output_dir / f"{model_name}_convergence.png"

    if not best_model_path.exists():
        msg = f"Model checkpoint not found: {best_model_path}"
        raise FileNotFoundError(msg)

    state_dict = torch.load(
        best_model_path,
        map_location=DEVICE,
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    model.eval()

    param_tensor = torch.as_tensor(
        test_params,
        dtype=torch.float32,
        device=DEVICE,
    ).unsqueeze(0)

    with torch.inference_mode():
        channel_amp_dense_batch = predict_chi_batch_from_params(
            forward_model=model,
            params_batch=param_tensor,
            coords=coords,
            Nx=nx,
            Ny=ny,
            Nz=nz,
        )

        # Shape: (2, Nx, Ny, Nz)
        channel_amp_dense = channel_amp_dense_batch[0]

        spatial_data_complex = torch.complex(
            channel_amp_dense[0],
            channel_amp_dense[1],
        ) / (nx * ny)

        k_space_complex = torch.fft.fft2(
            spatial_data_complex,
            dim=(0, 1),
        )

        predicted_state_data = k_space_complex.cpu().numpy()
    config = OptimizationConfig(precision=1e-5, max_iterations=1000, n_channels=160)
    condition = condition_from_params(test_params)
    actual_state_channels = get_stable_state(condition, config=config)

    actual_k_state_data = torch.complex(
        actual_state_channels[0],
        actual_state_channels[1],
    ).to(DEVICE)
    actual_real_state_data = torch.fft.ifft2(actual_k_state_data, dim=(0, 1))

    # ------------------------------------------------------------------
    # Central channel comparison
    # ------------------------------------------------------------------
    actual_psi_00 = actual_k_state_data[0, 0, :]
    predicted_psi_00 = predicted_state_data[0, 0, :]

    fig, ax = plot.get_figure()
    ax.set_xlabel("z")
    ax.set_ylabel(r"$\psi_{00}(z)$")
    ax.plot(
        z,
        actual_psi_00.real.detach().cpu().numpy(),
        label="Actual real part",
    )
    ax.plot(
        z,
        predicted_psi_00.real,
        label="Predicted real part",
    )
    ax.set_title("Actual and predicted scattering state")
    ax.legend()

    fig.savefig(
        output_dir / f"{model_name}_scattering_state.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close(fig)
    predicted_phase = np.angle(predicted_psi_00)
    actual_phase = np.angle(actual_psi_00.detach().cpu().numpy())

    fig, ax = plot.get_figure()
    ax.plot(z, actual_phase, label="Actual")
    ax.plot(z, predicted_phase, "--", label="Predicted")
    ax.set_xlabel("z")
    ax.set_ylabel("Phase (rad)")
    ax.set_title(r"Phase of $\psi_{00}(z)$")
    ax.legend()

    fig.savefig(output_dir / f"{model_name}_phase.png", dpi=300)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Channel-space slices (display in physical reciprocal-space order)
    # ------------------------------------------------------------------

    z_slices = [0, nz // 4, nz // 2, 3 * nz // 4, nz - 1]

    extent = (
        -nx // 2,
        nx // 2,
        -ny // 2,
        ny // 2,
    )

    for z_idx in z_slices:
        actual = np.fft.fftshift(
            actual_k_state_data[:, :, z_idx].abs().detach().cpu().numpy()
        )

        predicted = np.fft.fftshift(np.abs(predicted_state_data[:, :, z_idx]))

        error = np.abs(predicted - actual)

        vmax = max(actual.max(), predicted.max())

        fig, axes = plt.subplots(
            1,
            3,
            figsize=(15, 4),
            constrained_layout=True,
        )

        titles = [
            "Actual",
            "Predicted",
            "Absolute error",
        ]
        images = [
            actual,
            predicted,
            error,
        ]
        cmaps = [
            "viridis",
            "viridis",
            "inferno",
        ]

        for ax, img, title, cmap in zip(
            axes,
            images,
            titles,
            cmaps,
            strict=False,
        ):
            kwargs = {
                "origin": "lower",
                "extent": extent,
                "cmap": cmap,
                "interpolation": "nearest",
                "aspect": "equal",
            }

            if title != "Absolute error":
                kwargs["vmin"] = 0
                kwargs["vmax"] = vmax

            im = ax.imshow(img, **kwargs)

            ax.set_xlabel(r"Diffraction order $m$")
            ax.set_ylabel(r"Diffraction order $n$")
            ax.set_title(title)

            fig.colorbar(im, ax=ax)

        fig.suptitle(
            f"Channel-space amplitude (z index {z_idx})",
            fontsize=13,
        )

        fig.savefig(
            output_dir / f"{model_name}_channel_slice_z{z_idx:03d}.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)

    # ------------------------------------------------------------------
    # Scattering matrices
    # ------------------------------------------------------------------
    predicted_s_matrix_data = np.abs(predicted_state_data[:, :, -1]) ** 2
    predicted_s_matrix = Array(
        AsUpcast(basis.transformed_from_metadata(metadata_x01), metadata_x01),
        predicted_s_matrix_data.astype(np.complex128),
    )
    actual_s_matrix_data = (
        actual_k_state_data[:, :, -1].abs().square().detach().cpu().numpy()
    )
    actual_s_matrix = Array(
        AsUpcast(basis.transformed_from_metadata(metadata_x01), metadata_x01),
        actual_s_matrix_data.astype(np.complex128),
    )
    fig, ax, _mesh = plot.array_against_axes_2d_k_nearest_neighbor(
        actual_s_matrix,
        measure="abs",
    )
    ax.set_title("The actual scattering matrix")

    fig.savefig(
        output_dir / "scattering_matrix_actual.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close(fig)

    fig, ax, _mesh = plot.array_against_axes_2d_k_nearest_neighbor(
        predicted_s_matrix,
        measure="abs",
    )
    ax.set_title("The predicted scattering matrix")

    fig.savefig(
        output_dir / f"{model_name}_scattering_matrix_predicted.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close(fig)

    # ------------------------------------------------------------------
    # Real-space center-axis amplitude
    # ------------------------------------------------------------------
    center_x = nx // 2
    center_y = ny // 2

    predicted_psi_z = spatial_data_complex[
        center_x,
        center_y,
        :,
    ]
    actual_psi_z = actual_real_state_data[
        center_x,
        center_y,
        :,
    ]

    predicted_psi_z_abs = predicted_psi_z.abs().cpu().numpy()
    actual_psi_z_abs = actual_psi_z.abs().cpu().numpy()

    fig, ax = plot.get_figure()
    ax.set_xlabel("z", fontsize=11, fontweight="bold")
    ax.set_ylabel(
        r"$|\psi(x=0, y=0, z)|$ (Real Space)",
        fontsize=11,
        fontweight="bold",
    )

    ax.plot(
        z,
        actual_psi_z_abs,
        label="Actual amplitude",
        linewidth=2,
    )
    ax.plot(
        z,
        predicted_psi_z_abs,
        label="Predicted amplitude",
        linestyle="--",
        linewidth=2,
    )

    ax.set_title(
        "Real-space scattering amplitude along the center axis",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.legend(frameon=True)

    fig.savefig(
        output_dir / f"{model_name}_real_space_amplitude.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close(fig)

    print("--> Real-space amplitude plot saved successfully.")

    z_slices = [0, nz // 4, nz // 2, 3 * nz // 4, nz - 1]

    for z_idx in z_slices:
        plot_slice_pair(
            actual_3d=actual_real_state_data,
            predicted_3d=spatial_data_complex,
            axis="xy",
            idx=z_idx,
            output_path=output_dir / f"{model_name}_slice_xy_z{z_idx:03d}.png",
            title=f"XY slice at z index {z_idx}",
        )

    # ------------------------------------------------------------------
    # Regression metrics
    # ------------------------------------------------------------------
    def _metrics(
        predicted: np.ndarray | torch.Tensor,
        actual: np.ndarray | torch.Tensor,
        name: str,
    ) -> None:
        predicted_np = (
            predicted.detach().cpu().numpy()
            if isinstance(predicted, torch.Tensor)
            else np.asarray(predicted)
        )

        actual_np = (
            actual.detach().cpu().numpy()
            if isinstance(actual, torch.Tensor)
            else np.asarray(actual)
        )

        if predicted_np.shape != actual_np.shape:
            msg = (
                f"{name} shape mismatch: "
                f"predicted {predicted_np.shape}, "
                f"actual {actual_np.shape}."
            )
            raise ValueError(msg)

        difference = predicted_np - actual_np

        mse = float(np.mean(np.abs(difference) ** 2))
        rmse = float(np.sqrt(mse))

        actual_norm = np.linalg.norm(actual_np.ravel())
        predicted_norm = np.linalg.norm(predicted_np.ravel())

        relative_l2 = float(np.linalg.norm(difference.ravel()) / (actual_norm + 1e-12))

        if np.iscomplexobj(predicted_np) or np.iscomplexobj(actual_np):
            correlation = float(
                np.abs(
                    np.vdot(
                        actual_np.ravel(),
                        predicted_np.ravel(),
                    )
                )
                / (actual_norm * predicted_norm + 1e-12)
            )
            correlation_name = "CosSim"
        else:
            actual_std = np.std(actual_np)
            predicted_std = np.std(predicted_np)

            if actual_std < 1e-12 or predicted_std < 1e-12:
                correlation = float("nan")
            else:
                correlation = float(
                    np.corrcoef(
                        actual_np.ravel(),
                        predicted_np.ravel(),
                    )[0, 1]
                )

            correlation_name = "Corr"

        residual_sum = np.sum(np.abs(difference) ** 2)
        total_sum = np.sum(np.abs(actual_np - np.mean(actual_np)) ** 2) + 1e-12
        r_squared = float(1.0 - residual_sum / total_sum)

        print(f"\n[{name}]")
        print(f"MSE    : {mse:.6e}")
        print(f"RMSE   : {rmse:.6e}")
        print(f"RelL2  : {relative_l2:.6e}")
        print(f"{correlation_name:<7}: {correlation:.6f}")
        print(f"R^2    : {r_squared:.6f}")

    _metrics(
        spatial_data_complex,
        actual_real_state_data,
        name="Real space",
    )

    _metrics(
        predicted_state_data,
        actual_k_state_data,
        name="Channel space",
    )

    _metrics(
        predicted_s_matrix.raw_data.real,
        actual_s_matrix.raw_data.real,
        name="Scattering matrix",
    )

    # ------------------------------------------------------------------
    # Training convergence
    # ------------------------------------------------------------------
    if stats_path.exists():
        stats = TrainingStats.load(stats_path)

        fig, _ = plot_loss_curves(stats)
        fig.savefig(
            curve_path,
            bbox_inches="tight",
            dpi=300,
        )
        plt.close(fig)

        print(f"--> Training convergence plot saved to: {curve_path}")
    else:
        print(
            f"Training statistics not found at {stats_path}; skipping convergence plot."
        )


def plot_all_models_psi_00(
    model_zoo: list[ModelZooEntry],
    *,
    output_dir: Path = Path("data/15"),
) -> None:
    """Plot actual and predicted Re(psi_00(z)) for every model."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Construct the test condition
    # ------------------------------------------------------------------
    condition0 = MorseScatteringCondition(
        mass=HELIUM_MASS,
        morse_parameters=operator.build.CorrugatedMorseParameters(
            depth=7.63 * electron_volt * 10**-3,
            height=(1.0 / 1.1) * angstrom_si,
            offset=1.0 * angstrom_si,
            beta=0.05,
        ),
        metadata=scattering_metadata_from_stacked_delta_x(
            (
                np.array([4 * angstrom_si, 0, 0]),
                np.array([0, 4 * angstrom_si, 0]),
                np.array([0, 0, Z_HEIGHT * angstrom_si]),
            ),
            (15, 15, 200),
        ),
        incident_k=incident_k_from_angles(
            theta=np.deg2rad(0),
            phi=np.deg2rad(20),
            energy=HELIUM_ENERGY,
            mass=HELIUM_MASS,
        ),
    )

    test_params = params_from_condition(condition0)
    condition = condition_from_params(test_params)

    nx, ny, nz = condition.metadata.shape

    coords = _make_coords(
        nx,
        ny,
        nz,
        device=DEVICE,
        dtype=torch.float32,
    )

    _, metadata_z = split_scattering_metadata(condition.metadata)
    z = np.asarray(metadata_z.values)

    param_tensor = torch.as_tensor(
        test_params,
        dtype=torch.float32,
        device=DEVICE,
    ).unsqueeze(0)

    # ------------------------------------------------------------------
    # Compute the actual state once
    # ------------------------------------------------------------------
    actual_channels = simulate_preconditioned_state(test_params)

    actual_k_state = torch.complex(
        torch.as_tensor(actual_channels[0], device=DEVICE),
        torch.as_tensor(actual_channels[1], device=DEVICE),
    )

    if actual_k_state.shape != (nx, ny, nz):
        msg = (
            "Unexpected actual-state shape. "
            f"Expected {(nx, ny, nz)}, received "
            f"{tuple(actual_k_state.shape)}."
        )
        raise ValueError(msg)

    actual_psi_00 = actual_k_state[0, 0, :].detach().cpu().numpy()

    # ------------------------------------------------------------------
    # Evaluate all models
    # ------------------------------------------------------------------
    predictions: dict[str, np.ndarray] = {}

    for entry in model_zoo:
        model_name = entry.name
        checkpoint_path = entry.base_path / f"{model_name}_best.pth"

        if not checkpoint_path.is_file():
            print(f"Skipping {model_name}: checkpoint not found at {checkpoint_path}")
            continue

        model = entry.model.to(DEVICE)

        state_dict = torch.load(
            checkpoint_path,
            map_location=DEVICE,
            weights_only=True,
        )
        model.load_state_dict(state_dict)
        model.eval()

        with torch.inference_mode():
            dense_prediction = predict_chi_batch_from_params(
                forward_model=model,
                params_batch=param_tensor,
                coords=coords,
                Nx=nx,
                Ny=ny,
                Nz=nz,
            )

        # Expected shape: (1, 2, Nx, Ny, Nz)
        if dense_prediction.ndim != 5:
            msg = (
                f"{model_name} returned a tensor with shape "
                f"{tuple(dense_prediction.shape)}. Expected "
                "(batch, 2, Nx, Ny, Nz)."
            )
            raise ValueError(msg)

        if dense_prediction.shape[1] != 2:
            msg = (
                f"{model_name} returned {dense_prediction.shape[1]} "
                "output channels. Set output_dim=2 so that channel 0 "
                "is the real part and channel 1 is the imaginary part."
            )
            raise ValueError(msg)

        dense_prediction = dense_prediction[0]

        predicted_real_space = torch.complex(
            dense_prediction[0],
            dense_prediction[1],
        ) / (nx * ny)

        predicted_k_state = torch.fft.fft2(
            predicted_real_space,
            dim=(0, 1),
        )

        predicted_psi_00 = predicted_k_state[0, 0, :].detach().cpu().numpy()

        predictions[model_name] = predicted_psi_00

        # Release the GPU copy before moving to the next model.
        model.to("cpu")

    if not predictions:
        msg = (
            "No model checkpoints were found. Check each model entry's "
            "base_path and checkpoint filename."
        )
        raise FileNotFoundError(msg)

    # ------------------------------------------------------------------
    # Plot real parts
    # ------------------------------------------------------------------
    fig, ax = plot.get_figure()

    ax.plot(
        z,
        actual_psi_00.real,
        label="Actual",
        linewidth=3,
    )

    for model_name, predicted_psi_00 in predictions.items():
        ax.plot(
            z,
            predicted_psi_00.real,
            label=model_name,
            linewidth=1.8,
            linestyle="--",
        )

    ax.set_xlabel("z")
    ax.set_ylabel(r"$\mathrm{Re}\left[\psi_{00}(z)\right]$")
    ax.set_title(r"Actual and predicted $\psi_{00}(z)$ for all models")
    ax.legend()
    ax.grid(visible=True, alpha=0.25)

    figure_path = output_dir / "all_models_psi_00_real.png"

    fig.savefig(
        figure_path,
        bbox_inches="tight",
        dpi=300,
    )
    plt.close(fig)

    print(f"Saved model comparison plot to: {figure_path}")


if __name__ == "__main__":
    generate()

    model_zoo: list[ModelZooEntry] = [
        ModelZooEntry(
            name="PureSIREN",
            train=False,
            base_path=Path("data/example_network"),
            model=PureSIREN(
                param_dim=4, output_dim=2, first_omega_0=1.0, hidden_omega_0=1.0
            ),
        ).load_best(device=DEVICE),
    ]

    for m in model_zoo:
        if m.train:
            print(f"\n[{m.name}] Training started on device: {DEVICE}")
            train_model(model_entry=m, epochs=1000)

    for model in model_zoo:
        if model.stats_path.exists():
            stats = TrainingStats.load(model.stats_path)
            fig, ax = plot_loss_curves(stats)
            fig.savefig(model.base_path / model.name / "loss_curves.pdf")

    for entry in model_zoo:
        entry.base_path.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 60}")
        print(f"Model: {entry.name}")
        print(f"Train: {entry.train}")
        print(f"Path:  {entry.base_path}")
        print(f"{'=' * 60}")

        _test(
            model_name=entry.name,
            model=entry.model,
            output_dir=entry.base_path,
        )

    fig, _ = compare_model_validation_loss(model_zoo=model_zoo)
    fig.savefig("data/example_network/model_validation_loss_comparison.pdf")
    plot_all_models_psi_00(
        model_zoo,
        output_dir=Path("data/15"),
    )
