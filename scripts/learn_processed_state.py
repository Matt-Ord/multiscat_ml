import time
from dataclasses import dataclass
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
from multiscat.config import MorseScatteringCondition, momentum_from_angles
from multiscat.multiscat import (
    _as_natural_units,
    get_preconditioned_state_from_state,
    get_scattering_state,
)
from multiscat.multiscat._util import (
    get_a_wave_full_for_condition,
    get_b_wave_full_for_condition,
)
from scipy.constants import angstrom as angstrom_si  # type: ignore[import-untyped]
from scipy.constants import (  # type: ignore[import-untyped]
    atomic_mass,
    electron_volt,
    physical_constants,
)
from slate_core import (
    Array,
    basis,
    metadata,
    plot,
)
from slate_core.basis import AsUpcast
from slate_core.metadata import (
    LobattoSpacedLengthMetadata,
)
from slate_core.metadata._spaced import Domain
from slate_quantum import operator
from torch import nn, optim
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from tqdm import tqdm

from multiscat_ml import plot_loss_curves
from multiscat_ml.utils import (
    TrainingStats,
)

# Constants
HELIUM_MASS = physical_constants["alpha particle mass"][0]
HELIUM_ENERGY = 7 * electron_volt * 10**-3
Z_HEIGHT = 8
Nx, Ny, Nz = 15, 15, 100


@dataclass(kw_only=True, frozen=True)
class ModelZooEntry:
    """Entry in model_zoo containing training/loading flags, base path, and PyTorch model."""

    train: bool = False
    load: bool = True
    base_path: Path
    model: nn.Module


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")  # pyright: ignore[reportConstantRedefinition]
else:
    DEVICE = torch.device("cpu")  # pyright: ignore[reportConstantRedefinition]

PARAMS_MIN = np.array(
    [0.5, 5.0, 0.5, 0.0, 0.02, 6, 0.0, 0.0, 2, 3],
    dtype=np.float64,
)
PARAMS_MAX = np.array(
    [4, 10.0, 1.5, 4.0, 0.20, 10, np.pi / 2, 2 * np.pi, 40, 10],
    dtype=np.float64,
)


def denormalize_params(
    params_norm: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    """Scales parameters back to their original physical units."""
    return params_norm * (PARAMS_MAX - PARAMS_MIN) + PARAMS_MIN


def condition_from_params(
    params: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> MorseScatteringCondition:
    """Convert a tensor of parameters into a ScatteringCondition."""
    (
        a,
        depth,
        height,
        offset,
        beta,
        z_height,
        theta,
        phi,
        energy,
        mass,
    ) = denormalize_params(
        params,
    )

    morse_params = operator.build.CorrugatedMorseParameters(
        depth=depth * electron_volt * 10**-3,
        height=height * angstrom_si,
        offset=offset * angstrom_si,
        beta=beta,
    )

    metadata = scattering_metadata_from_stacked_delta_x(
        (
            np.array([a * angstrom_si, 0, 0]),
            np.array([0, a * angstrom_si, 0]),
            np.array([0, 0, z_height * angstrom_si]),
        ),
        (Nx, Ny, Nz),
    )

    return MorseScatteringCondition(
        mass=mass * atomic_mass,
        morse_parameters=morse_params,
        metadata=metadata,
        incident_k=momentum_from_angles(
            theta=theta,
            phi=phi,
            energy=energy * electron_volt * 10**-3,
            mass=mass * atomic_mass,
        ),
    )


def normalize_params(
    params: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    """Scales parameters to a [0, 1] range."""
    return (params - PARAMS_MIN) / (PARAMS_MAX - PARAMS_MIN)


def params_from_condition(
    condition: MorseScatteringCondition,
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    """Extract the parameters from a ScatteringCondition."""
    metadata_x01, metadata_z = split_scattering_metadata(condition.metadata)
    z_height = metadata_z.domain.delta / angstrom_si

    a = metadata.volume.fundamental_stacked_delta_x(metadata_x01)[0][0] / angstrom_si

    morse_parameters = condition.morse_parameters

    depth = morse_parameters.depth / (electron_volt * 10**-3)
    height = morse_parameters.height / angstrom_si
    offset = morse_parameters.offset / angstrom_si
    beta = morse_parameters.beta

    theta = condition.theta
    phi = condition.phi
    energy = condition.incident_energy / (electron_volt * 10**-3)
    mass = condition.mass / atomic_mass

    return normalize_params(
        np.array(
            [
                a,
                depth,
                height,
                offset,
                beta,
                z_height,
                theta,
                phi,
                energy,
                mass,
            ],
        ),
    )


def simulate_preconditioned_state_from_condition(
    condition: MorseScatteringCondition,
) -> torch.Tensor:
    """Simulate the preconditioned state from the given ScatteringCondition."""
    config = OptimizationConfig(precision=1e-5, max_iterations=1000, n_channels=625)
    morse_parameters = condition.morse_parameters
    _offset, _depth = morse_parameters.offset, morse_parameters.depth
    converted_condition = _as_natural_units(condition)
    b = get_b_wave_full_for_condition(
        converted_condition.metadata, converted_condition.incident_k
    )

    a_full = get_a_wave_full_for_condition(
        converted_condition.metadata, converted_condition.incident_k
    )

    a = np.zeros(converted_condition.metadata.shape, dtype=np.complex128)
    a[0, 0, :] = a_full[0, 0, :]

    # Get the scattering state
    state = get_scattering_state(condition, config)

    # Get the preconditioned state
    preconditioned_state = get_preconditioned_state_from_state(
        state, condition, n_channels=config.n_channels
    )

    # Return the real and imaginary parts of the preconditioned state
    preconditioned_data = preconditioned_state.with_basis(
        close_coupling_basis(condition.metadata)
    ).raw_data.reshape(condition.metadata.shape)
    stabilized = a + 2.0j * b * preconditioned_data

    return torch.from_numpy(
        np.stack((stabilized.real, stabilized.imag), axis=0)
    ).float()


def simulate_preconditioned_state(
    params: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> torch.Tensor:
    """Simulate the preconditioned state from the given parameters."""
    condition = condition_from_params(params)
    return simulate_preconditioned_state_from_condition(condition)


def generate_dataset_hdf5(filepath: Path, num_samples: int = 500) -> None:
    """Generate parameters and Preconditioned state, saving them directly to disk."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if filepath.exists():
        print(f"Dataset already exists at {filepath}. Skipping generation.")
        return

    print(
        f"Generating {num_samples} samples straight to disk. This may take a while...",
    )
    rng = np.random.default_rng()

    # Open an HDF5 file in write mode
    with h5py.File(filepath, "w") as f:
        # Fixed-size inputs
        x_ds = f.create_dataset("X", shape=(num_samples, 10), dtype=np.float64)

        # Variable-size outputs go in a group
        y_ds = f.create_dataset(
            "Y", shape=(num_samples, 2, Nx, Ny, Nz), dtype=np.float32
        )
        for i in range(num_samples):
            print(f"Generating sample {i + 1}/{num_samples}")

            params = rng.uniform(size=10)
            x_ds[i] = params
            y_ds[i] = simulate_preconditioned_state(params)


class HDF5ScatteringDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """An optimized Dataset that preloads scattering data completely into RAM."""

    def __init__(self, filepath: Path) -> None:
        self.filepath = filepath

        # Open, read everything into RAM instantly, and close immediately
        with h5py.File(name=filepath, mode="r") as f:
            print(f"--> Preloading {filepath.name} entirely into system RAM...")
            # Loading full arrays into memory
            X_raw = torch.from_numpy(f["X"][:]).float()
            Y_raw = torch.from_numpy(f["Y"][:]).float()

        Y_complex = torch.complex(
            Y_raw[:, 0, ...], Y_raw[:, 1, ...]
        )  # Shape: (B, Nx, Ny, Nz)

        print("--> Computing 2D Fourier Transform over the x-y plane...")
        # 3. Perform 2D FFT over the kx (dim=1) and ky (dim=2) dimensions
        # We shift low frequencies to the center using fftshift for physical correctness
        Y_ifft = (
            torch.fft.ifft2(Y_complex, s=(Nx * 5, Ny * 5), dim=(1, 2)) * Nx * Ny * 25
        )

        # 4. Pack it back into a split Real/Imaginary view if your SIREN model expects 2 channels
        self.Y_data = torch.stack(
            [Y_ifft.real, Y_ifft.imag], dim=1
        )  # Shape: (B, 2, Nx, Ny, Nz)
        self.X_data = X_raw

        self.length = self.X_data.shape[0]
        print(f"--> Caching complete! Loaded {self.length} samples.")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        # Fast RAM slice—no disk reading overhead!
        return (
            self.X_data[index],
            self.Y_data[index],
        )


def generate() -> None:
    for i in range(50):
        data_path = Path(f"data/15/env_stabilized_data_{i}.hdf5")
        generate_dataset_hdf5(data_path, num_samples=500)


def load_datasets() -> ConcatDataset[tuple[torch.Tensor, torch.Tensor]]:
    datasets = [
        HDF5ScatteringDataset(Path(f"data/15/env_stabilized_data_{i}.hdf5"))
        for i in range(50)
    ]
    return ConcatDataset[tuple[torch.Tensor, torch.Tensor]](datasets)


class _SineLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        is_first: bool = False,
        omega_0: float = 30.0,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.is_first = is_first
        self.omega_0 = omega_0
        self.init_weights()

    def init_weights(self) -> None:
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / self.linear.in_features
            else:
                bound = np.sqrt(6.0 / self.linear.in_features) / self.omega_0

            self.linear.weight.uniform_(-bound, bound)
            if self.linear.bias is not None:
                self.linear.bias.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


def align_params_to_coords(
    params: torch.Tensor,
    coords: torch.Tensor,
    *,
    param_dim: int,
    coord_dim: int,
) -> torch.Tensor:
    """
    Align parameter rows with coordinate rows.

    Supported shapes:
        params: (N, param_dim), coords: (N, coord_dim)
        params: (1, param_dim), coords: (N, coord_dim)
        params: (B, param_dim), coords: (B * N_pts, coord_dim)

    For flattened batched coordinates, points must be grouped by sample.
    """
    if params.ndim != 2:
        msg = f"params must be 2D, but received shape {tuple(params.shape)}."
        raise ValueError(msg)

    if coords.ndim != 2:
        msg = f"coords must be 2D, but received shape {tuple(coords.shape)}."
        raise ValueError(msg)

    if params.shape[-1] != param_dim:
        msg = (
            f"Expected params.shape[-1] == {param_dim}, "
            f"but received {params.shape[-1]}."
        )
        raise ValueError(msg)

    if coords.shape[-1] != coord_dim:
        msg = (
            f"Expected coords.shape[-1] == {coord_dim}, "
            f"but received {coords.shape[-1]}."
        )
        raise ValueError(msg)

    num_param_rows = params.shape[0]
    num_coord_rows = coords.shape[0]

    if num_param_rows == num_coord_rows:
        return params

    if num_param_rows == 1:
        return params.expand(num_coord_rows, -1)

    if num_coord_rows % num_param_rows == 0:
        points_per_sample = num_coord_rows // num_param_rows
        return params.repeat_interleave(points_per_sample, dim=0)

    msg = (
        "Could not align params and coords. "
        f"Received {num_param_rows} parameter rows and "
        f"{num_coord_rows} coordinate rows."
    )
    raise ValueError(msg)


class ExplicitAsymptoticSirenNet(nn.Module):
    """
    Persistent + Transient SIREN representation.

    Learns

        f(params, x, y, z)
            = persistent(params, x, y)
            + gate(z, params) * transient(params, x, y, z)

    where

        persistent -> a
        transient * gate  -> 2ib^T (Y-c)^-1 b
        gate       -> smooth transition into the asymptotic regime
    """

    def __init__(
        self,
        param_dim: int = 10,
        coord_dim: int = 3,
        hidden_dim: int = 64,
        transient_hidden_dim: int = 64,
        gate_hidden_dim: int = 64,
        output_dim: int = 2,
        omega_0: float = 10.0,
    ) -> None:
        super().__init__()

        if coord_dim != 3:
            msg = "coord_dim must be 3."
            raise ValueError(msg)

        self.param_dim = param_dim
        self.coord_dim = coord_dim
        self.output_dim = output_dim

        # -------------------------
        # Persistent branch
        # -------------------------
        self.persistent_net = nn.Sequential(
            _SineLayer(
                in_features=2 + param_dim,
                out_features=hidden_dim,
                is_first=True,
                omega_0=omega_0,
            ),
            nn.Linear(hidden_dim, output_dim),
        )

        # -------------------------
        # Transient branch
        # -------------------------
        self.transient_net = nn.Sequential(
            _SineLayer(
                in_features=coord_dim + param_dim,
                out_features=transient_hidden_dim,
                is_first=True,
                omega_0=omega_0,
            ),
            _SineLayer(
                transient_hidden_dim,
                transient_hidden_dim,
                is_first=False,
                omega_0=omega_0,
            ),
            nn.Linear(transient_hidden_dim, output_dim),
        )

        # -------------------------
        # Transition gate
        # -------------------------
        self.z_gate = nn.Sequential(
            nn.Linear(1 + param_dim, gate_hidden_dim),
            nn.GELU(),
            nn.Linear(gate_hidden_dim, 1),
        )

    @override
    def forward(
        self,
        params: torch.Tensor,
        coords: torch.Tensor,
    ) -> torch.Tensor:

        params = align_params_to_coords(
            params,
            coords,
            param_dim=self.param_dim,
            coord_dim=self.coord_dim,
        )

        xy = coords[:, :2]
        z = coords[:, 2:3]

        # Persistent asymptotic component
        persistent_inputs = torch.cat([xy, params], dim=-1)
        y_persistent = self.persistent_net(persistent_inputs)

        # Transient component
        transient_inputs = torch.cat([coords, params], dim=-1)
        y_transient = self.transient_net(transient_inputs)

        # Learned transition
        gate_inputs = torch.cat([z, params], dim=-1)
        gate = torch.sigmoid(self.z_gate(gate_inputs))

        return y_persistent + gate * y_transient


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
    model: nn.Module,
    model_name: str,
    epochs: int = 6000,
    max_epochs_without_improvement: int = 360,
    output_dir: Path = Path("data/15"),
) -> None:

    output_dir.mkdir(parents=True, exist_ok=True)

    model = model.to(DEVICE)
    # noqa:
    forward_criterion = nn.MSELoss()
    if isinstance(forward_criterion, nn.Module):
        forward_criterion = forward_criterion.to(DEVICE)

    coord_ext = _make_coords(
        Nx * 5, Ny * 5, Nz, device=DEVICE, dtype=torch.float32
    ).view(5 * Nx, 5 * Ny, Nz, 3)

    forward_optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-5,
    )

    scheduler_f = optim.lr_scheduler.ReduceLROnPlateau(
        forward_optimizer,
        mode="min",
        factor=0.5,
        patience=5,
    )

    dataset = load_datasets()
    train_dataset, val_dataset = random_split(dataset, [0.8, 0.2])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    stats_path = output_dir / f"{model_name}_training_stats.pkl"
    best_model_path = output_dir / f"{model_name}_best.pth"
    final_model_path = output_dir / f"{model_name}_final.pth"
    curve_path = output_dir / f"{model_name}_convergence.png"

    stats = TrainingStats()

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    print(
        f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples..."
    )
    print(f"Using device: {DEVICE}")

    val_sx = slice(2 * Nx, 3 * Nx)
    val_sy = slice(2 * Ny, 3 * Ny)

    for epoch in range(epochs):
        # Ramp alpha from 0 to 1 after switch_epoch.
        model.train()
        forward_criterion.train()

        # ------------------------------------------------------------------
        # Training
        # ------------------------------------------------------------------
        train_loss = 0.0

        train_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{epochs}",
            unit="batch",
        )

        for batch_idx, (
            params_batch,
            target_batch,
        ) in enumerate(train_bar):
            forward_optimizer.zero_grad(set_to_none=True)

            t0 = time.perf_counter()
            sx, sy = sample_crop(
                Nx_ext=5 * Nx,
                Ny_ext=5 * Ny,
                Nx=Nx,
                Ny=Ny,
                device=DEVICE,
            )

            coord = coord_ext[sx, sy].reshape(-1, 3)

            target = target_batch[:, :, sx, sy, :]

            prediction = predict_chi_batch_from_params(
                forward_model=model,
                params_batch=params_batch,
                coords=coord,
                Nx=Nx,
                Ny=Ny,
                Nz=Nz,
            )

            t1 = time.perf_counter()

            loss_all = forward_criterion(prediction, target)

            loss = loss_all
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            t2 = time.perf_counter()

            forward_optimizer.step()

            t3 = time.perf_counter()

            train_loss += loss.item()
            average_batch_loss = train_loss / (batch_idx + 1)

            train_bar.set_postfix(
                loss=f"{loss.item():.3e}",
                avg=f"{average_batch_loss:.3e}",
                pred=f"{t1 - t0:.2f}s",
                back=f"{t2 - t1:.2f}s",
                step=f"{t3 - t2:.2f}s",
                lr=f"{forward_optimizer.param_groups[0]['lr']:.1e}",
            )

        average_train_loss = train_loss / len(train_loader)

        # ------------------------------------------------------------------
        # Validation
        # ------------------------------------------------------------------
        model.eval()
        forward_criterion.eval()
        val_loss = 0.0

        val_bar = tqdm(
            val_loader,
            desc=f"Val {epoch + 1}/{epochs}",
            unit="batch",
        )

        with torch.inference_mode():
            for batch_idx, (
                params_batch,
                target_batch,
            ) in enumerate(val_bar):
                coord = coord_ext[val_sx, val_sy].reshape(-1, 3)

                target = target_batch[:, :, val_sx, val_sy, :]

                prediction = predict_chi_batch_from_params(
                    forward_model=model,
                    params_batch=params_batch,
                    coords=coord,
                    Nx=Nx,
                    Ny=Ny,
                    Nz=Nz,
                )

                loss_all = forward_criterion(prediction, target)
                batch_val_loss = (loss_all).item()

                val_loss += batch_val_loss
                average_batch_val_loss = val_loss / (batch_idx + 1)

                val_bar.set_postfix(
                    loss=f"{batch_val_loss:.3e}",
                    avg=f"{average_batch_val_loss:.3e}",
                )

        average_val_loss = val_loss / len(val_loader)

        # ------------------------------------------------------------------
        # Epoch bookkeeping
        # ------------------------------------------------------------------
        current_weight_decay = forward_optimizer.param_groups[0].get(
            "weight_decay",
            0.0,
        )

        stats.append(
            train_loss=average_train_loss,
            val_loss=average_val_loss,
            weight_decay=current_weight_decay,
        )
        stats.save(stats_path)

        scheduler_f.step(average_val_loss)

        current_lr = forward_optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch + 1:03d}/{epochs} | "
            f"Fwd Loss (Tr/Val): "
            f"{average_train_loss:.2e} / {average_val_loss:.2e} | "
            f"LR: {current_lr:.1e}"
        )
        # ------------------------------------------------------------------
        # Best checkpoint and early stopping
        # ------------------------------------------------------------------
        if average_val_loss < best_val_loss:
            best_val_loss = average_val_loss
            epochs_without_improvement = 0

            torch.save(
                model.state_dict(),
                best_model_path,
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= max_epochs_without_improvement:
            print(
                "Early stopping triggered after "
                f"{epochs_without_improvement} epochs without improvement."
            )
            break

    # ----------------------------------------------------------------------
    # Final outputs
    # ----------------------------------------------------------------------
    torch.save(
        model.state_dict(),
        final_model_path,
    )

    stats.save(stats_path)

    fig, _ = plot_loss_curves(stats)
    fig.savefig(
        curve_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    print("Training complete.")
    print(f"Best validation loss: {best_val_loss:.3e}")
    print(f"Saved training statistics to: {stats_path}")
    print(f"Saved convergence curve to: {curve_path}")
    print(f"Saved final model to: {final_model_path}")


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
        incident_k=momentum_from_angles(
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

    actual_state_channels = simulate_preconditioned_state(test_params)

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
    model_zoo: dict[str, ModelZooEntry],
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
        incident_k=momentum_from_angles(
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

    for model_name, entry in model_zoo.items():
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
    param_dim = 10
    coord_dim = 3
    output_dim = 2
    generate()

    model_zoo: dict[str, ModelZooEntry] = {
        "ExplicitAsymptoticSirenNet": ModelZooEntry(
            model=ExplicitAsymptoticSirenNet(
                param_dim=param_dim,
                coord_dim=coord_dim,
                hidden_dim=16,
                omega_0=30.0,
                output_dim=output_dim,
            ),
            base_path=Path("data/15/ExplicitAsymptoticSirenNet"),
            train=False,
            load=False,
        ),
    }

    for model_name, entry in model_zoo.items():
        entry.base_path.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 60}")
        print(f"Model: {model_name}")
        print(f"Train: {entry.train}")
        print(f"Load:  {entry.load}")
        print(f"Path:  {entry.base_path}")
        print(f"{'=' * 60}")

        if entry.train:
            train_model(
                model=entry.model,
                model_name=model_name,
                output_dir=entry.base_path,
                epochs=1000,
                max_epochs_without_improvement=200,
            )

        if entry.load:
            _test(
                model_name=model_name,
                model=entry.model,
                output_dir=entry.base_path,
            )
    plot_all_models_psi_00(
        model_zoo,
        output_dir=Path("data/15"),
    )
