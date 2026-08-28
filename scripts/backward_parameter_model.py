import contextlib
from collections.abc import Callable
from pathlib import Path
from typing import Any, override

import h5py  # type: ignore[import-untyped]
import matplotlib.pyplot as plt
import numpy as np
import torch
from multiscat import OptimizationConfig, ScatteringCondition, get_scattering_matrix
from multiscat.basis import (
    as_scattering_potential,
    scattering_metadata_from_stacked_delta_x,
    split_scattering_metadata,
)
from scipy.constants import angstrom as angstrom_si  # type: ignore[import-untyped]
from scipy.constants import (  # type: ignore[import-untyped]
    electron_volt,
    physical_constants,
)
from slate_core import AsUpcast, basis
from slate_quantum import operator
from torch import nn, optim
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from tqdm import tqdm

from multiscat_ml.utils import plot_training_convergence, save_loss_history

# Constants
HELIUM_MASS = physical_constants["alpha particle mass"][0]
HELIUM_ENERGY = 20 * electron_volt * 10**-3
Z_HEIGHT = 8

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")  # pyright: ignore[reportConstantRedefinition]
else:
    DEVICE = torch.device("cpu")  # pyright: ignore[reportConstantRedefinition]

PARAMS_MIN = np.array([5.0, 0.5, 0.05, 2.5, 0.0], dtype=np.float64)
PARAMS_MAX = np.array([15.0, 1.2, 0.20, 4.0, 2 * np.pi / 5], dtype=np.float64)
OFFSET = 1.0
Nx, Ny, Nz = 11, 11, 100
potential_channels = 11
potential_z = 100
z = torch.linspace(0.0, Z_HEIGHT, potential_z)


def denormalize_params(
    params_norm: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    """Scales parameters back to their original physical units."""
    return params_norm * (PARAMS_MAX - PARAMS_MIN) + PARAMS_MIN


def corrugated_morse_real_space(  # ruff: ignore[too-many-arguments]  # ruff: ignore[too-many-positional-arguments]
    depth: float,
    height: float,
    offset: float,
    beta: float,
    x: torch.Tensor | np.ndarray,
    y: torch.Tensor | np.ndarray,
    z: torch.Tensor | np.ndarray,
    a: float,
) -> torch.Tensor:
    """Return (n_channels, n_channels, nz)."""
    z = torch.as_tensor(z, device=DEVICE, dtype=torch.float32)
    x = torch.as_tensor(x, device=DEVICE, dtype=torch.float32)
    y = torch.as_tensor(y, device=DEVICE, dtype=torch.float32)

    t = torch.exp(-(z - offset) / height)
    v0 = depth * (t**2 - 2 * t)
    v1 = -2 * beta * depth * t**2

    q = torch.cos(2 * np.pi * x / a) + torch.cos(2 * np.pi * y / a)

    return v0 + (v1 * q)


def condition_from_params(
    params: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> ScatteringCondition:
    """Convert an array of normalised parameters into a ScatteringCondition."""
    depth, height, beta, unit_cell, theta = denormalize_params(params)

    # Everything below is in SI, matching corrugated_morse_real_space's convention.
    depth_si = depth * electron_volt * 10**-3
    height_si = height * angstrom_si
    offset_si = OFFSET * angstrom_si
    a_si = unit_cell * angstrom_si

    metadata = scattering_metadata_from_stacked_delta_x(
        (
            np.array([a_si, 0, 0]),
            np.array([0, a_si, 0]),
            np.array([0, 0, Z_HEIGHT * angstrom_si]),
        ),
        (Nx, Ny, Nz),
    )

    initial_potential = operator.build.potential_from_function(
        metadata,
        lambda x: (
            corrugated_morse_real_space(
                depth=depth_si,
                height=height_si,
                offset=offset_si,
                beta=beta,
                x=x[0],
                y=x[1],
                z=x[2],
                a=a_si,
            )
            .detach()
            .cpu()
            .numpy()
            .astype(np.complex128)
        ),
    )
    potential = as_scattering_potential(initial_potential, metadata)

    return ScatteringCondition.from_angles(
        mass=HELIUM_MASS,
        energy=HELIUM_ENERGY,
        theta=theta,
        phi=np.deg2rad(0),
        potential=potential,
    )


def normalize_params(
    params: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    """Scales parameters to a [0, 1] range."""
    return (params - PARAMS_MIN) / (PARAMS_MAX - PARAMS_MIN)


def simulate_s_matrix(
    params: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    """Wrap your physics code into a single callable function."""
    condition = condition_from_params(params)

    config = OptimizationConfig(precision=1e-5, max_iterations=1000000, n_channels=150)
    s_matrix = get_scattering_matrix(condition, config, backend="scipy")

    metadata_x01, _ = split_scattering_metadata(condition.metadata)
    return s_matrix.with_basis(
        AsUpcast(basis.transformed_from_metadata(metadata_x01), metadata_x01),
    ).raw_data.real.reshape(Nx, Ny)


def corrugated_morse_channels(  # ruff: ignore[too-many-arguments]
    z: torch.Tensor | np.ndarray,
    *,
    depth: float,
    height: float,
    offset: float,
    beta: float,
    n_channels: int = potential_channels,
) -> torch.Tensor:
    """Return potential in the shape (n_channels, n_channels, nz)."""
    z = torch.as_tensor(z, dtype=torch.float32)

    t = torch.exp(-(z - offset) / height)
    v0 = depth * (t**2 - 2 * t)
    v1 = -beta * depth * t**2

    v = torch.zeros(
        (n_channels, n_channels, z.shape[0]), dtype=z.dtype, device=z.device
    )

    v[0, 0] = v0
    for m, n in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        v[m, n] = v1

    return v


def get_morse_potential(
    params: torch.Tensor | np.ndarray,
    nz: int,
    n_channels: int = potential_channels,
) -> torch.Tensor:
    if isinstance(params, torch.Tensor):
        params = params.detach().cpu().numpy()
    depth, height, offset, beta, *_ = denormalize_params(
        np.asarray(params, dtype=np.float64)
    )
    z = np.linspace(0.0, Z_HEIGHT, nz)
    return corrugated_morse_channels(
        z,
        depth=float(depth),
        height=float(height),
        offset=float(offset),
        beta=float(beta),
        n_channels=n_channels,
    )


class CoordMLP(nn.Module):
    """Maps coordinates to a feature vector, applied pointwise over the last axis."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    @override
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.net(coords)


class ResBlock(nn.Module):
    """Pre-activation residual block: the skip path is a clean identity."""

    def __init__(self, hidden_dim: int, dropout_rate: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout_rate),
        )

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


def _make_s_coords(
    nx: int,
    ny: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Generate the channel indices in FFT-style orders.

    This will be fed to the CoordMLP for the index mapping required for a flexible S -> params model.
    """
    xs = torch.fft.fftfreq(nx, d=1.0 / nx).to(device=device, dtype=dtype)
    ys = torch.fft.fftfreq(ny, d=1.0 / ny).to(device=device, dtype=dtype)

    # Create the 2D grid
    grid = torch.stack(
        torch.meshgrid(xs, ys, indexing="ij"),
        dim=-1,
    )  # (Nx, Ny, 2)

    return grid.reshape(-1, 2)  # (N, 2)


class JointKernelBackwardNet(nn.Module):
    def __init__(  # ruff: ignore[too-many-arguments]
        self,
        s_idx_dim: int = 2,
        encoder_dim: int = 32,
        hidden_dim: int = 256,
        n_blocks: int = 4,
        dropout_rate: float = 0.05,
        *,
        use_encoder: bool = True,
    ) -> None:
        super().__init__()
        self.encoder_dim = encoder_dim
        self.use_encoder = use_encoder

        # The optional encoder: same linear-functional-of-S construction as
        # before, u_d(n, m) generated from the channel coordinate.
        self.encoder = CoordMLP(s_idx_dim, encoder_dim) if use_encoder else None

        cond_dim = encoder_dim if use_encoder else 0

        # Split embedding again, for the same reason as in the separable model:
        # the (n, m, n', m', z) term has no batch axis, so computing it once.
        self.embed_coords = nn.Linear(s_idx_dim, hidden_dim, bias=True)
        self.embed_cond = (
            nn.Linear(cond_dim, hidden_dim, bias=False) if cond_dim else None
        )
        self.embed_norm = nn.Sequential(nn.LayerNorm(hidden_dim), nn.GELU())

        self.res_blocks = nn.Sequential(
            *(ResBlock(hidden_dim, dropout_rate) for _ in range(n_blocks))
        )
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 5))

    def encode(self, s_coords: torch.Tensor, s_values: torch.Tensor) -> torch.Tensor:
        """
        (N, 2), (B, N) -> (B, encoder_dim).

        Encoder (global features) is fed to the index mapping for nonlinearity.
        """
        kernel = self.encoder(s_coords)  # ty: ignore[call-non-callable]
        return torch.einsum("nd,bn->bd", kernel, s_values)

    def kernel(
        self,
        s_coords: torch.Tensor,  # (N, 2)
        cond: torch.Tensor | None,  # (B, cond_dim) or None
    ) -> torch.Tensor:
        """
        Return C with shape (B, N, 5), or (1, N, 5) when cond is None.

        This is the resultant index mapping used to build the S -> params tensor, features of S is mixed in the tensor via the learnt encoder,
        so this map is nonlinear.
        """
        h = self.embed_coords(s_coords).unsqueeze(0)  # (1, N, H)
        if cond is not None and self.embed_cond is not None:
            h = h + self.embed_cond(cond)[:, None, :]  # noqa: PLR6104

        x = self.embed_norm(h)
        x = self.res_blocks(x)
        return self.head(x)  # (B, N, 5)

    @override
    def forward(
        self,
        s_coords: torch.Tensor,  # (N, 2)
        s_values: torch.Tensor,  # (B, N) -- |S|^2
    ) -> torch.Tensor:
        """Return (B, 5) -- depth, height, beta, unit_cell, theta."""
        parts: list[torch.Tensor] = []
        if self.use_encoder:
            parts.append(self.encode(s_coords, s_values))
        cond = torch.cat(parts, dim=-1) if parts else None

        c = self.kernel(s_coords, cond)  # (B, N, 5) or (1, N, 5)
        return torch.einsum("bnk,bn->bk", c.expand(s_values.shape[0], -1, -1), s_values)


@torch.no_grad()
def predict_potential_params(
    model: JointKernelBackwardNet,
    s_matrix: torch.Tensor,  # (B, Nx, Ny) -- |S|^2
    nx: int,
    ny: int,
) -> torch.Tensor:
    """Parameter regression. Returns (B, 5)."""
    device, dtype = s_matrix.device, s_matrix.dtype
    s_coords = _make_s_coords(nx, ny, device, dtype)
    s_flat = s_matrix.reshape(s_matrix.shape[0], -1)

    return model(s_coords, s_flat)


def _make_target_potential_batch(
    params_batch: torch.Tensor,  # (B, n_params)
    n_channels: int,
    nz: int,
    potential_fn: Callable[..., torch.Tensor] = get_morse_potential,
) -> torch.Tensor:
    return torch.stack(
        [potential_fn(p, n_channels=n_channels, nz=nz) for p in params_batch], dim=0
    ).to(device=params_batch.device, dtype=params_batch.dtype)


def generate_dataset_hdf5(filepath: Path, num_samples: int = 1000) -> None:
    """Generate parameters and S-matrices, saving them directly to disk."""
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
        # Create empty datasets on disk
        input_data = f.create_dataset("X", shape=(num_samples, 5), dtype=np.float64)  # type: ignore[hd5]
        s_data = f.create_dataset(  # type: ignore[hd5]
            "S",
            shape=(num_samples, 11, 11),
            dtype=np.float64,
        )

        for i in range(num_samples):
            print(f"Generating sample {i + 1}/{num_samples}")
            params = rng.uniform(size=5)

            input_data[i] = params
            s_data[i] = simulate_s_matrix(params)


class HDF5ScatteringDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """A PyTorch Dataset that reads scattering data from an HDF5 file on demand."""

    def __init__(self, filepath: Path) -> None:
        self.filepath = filepath
        self.file = None

        try:
            with h5py.File(filepath, "r") as f:
                self.length: int = f["X"].shape[0]  # type: ignore[hd5]
        except FileNotFoundError:
            self.length = 0

    def __len__(self) -> int:
        """Get the length."""
        return self.length

    @override
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:  # ty:ignore[invalid-method-override]
        # Lazy initialization of the HDF5 file handler.
        # This is best practice to avoid errors if using multiple DataLoader workers.
        if self.file is None:
            self.file = h5py.File(self.filepath, "r")

        x_tensor = torch.tensor(self.file["X"][idx]).float()  # type: ignore[untyped]
        s_tensor = torch.tensor(self.file["S"][idx]).float()  # type: ignore[untyped]

        return x_tensor, s_tensor

    def __del__(self) -> None:
        if self.file is not None:
            self.file.close()


@contextlib.contextmanager
def freeze_parameters(model: nn.Module) -> Any:  # noqa: ANN401
    """Temporarily disables gradient computation for a model's parameters."""
    # Save the original requires_grad state for each parameter
    original_states = {param: param.requires_grad for param in model.parameters()}

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    try:
        yield  # This is where the code inside your `with` block runs
    finally:
        # Restore the original states afterward, even if an error occurs
        for param, original_state in original_states.items():
            param.requires_grad = original_state


def generate() -> None:
    for i in range(10, 15):
        data_path = Path(f"data/backward_model/training_data_5params/dataset.{i}.hdf5")
        generate_dataset_hdf5(data_path, num_samples=1000)


def load_datasets() -> ConcatDataset[tuple[torch.Tensor, torch.Tensor]]:
    datasets = [
        HDF5ScatteringDataset(
            Path(f"data/backward_model/training_data_5params/dataset.{i}.hdf5")
        )
        for i in range(20)
    ]
    return ConcatDataset[tuple[torch.Tensor, torch.Tensor]](datasets)


def flat_channel_index(nx: int, ny: int, nz: int, device: torch.device) -> torch.Tensor:
    """(P,) mapping each flattened query point to its (n, m) channel id."""
    return torch.arange(nx * ny * nz, device=device) // nz


def train(  # noqa: PLR0914 # ruff: ignore[too-many-arguments]  # ruff: ignore[too-many-statements]
    device: torch.device = DEVICE,
    nx: int = Nx,
    ny: int = Ny,
    nz: int = Nz,
    *,
    batch_size: int = 512,
    epochs: int = 2000,
) -> None:
    dataset = load_datasets()
    train_dataset, val_dataset = random_split(dataset, [0.8, 0.2])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = JointKernelBackwardNet().to(device)

    # Coordinates are grid geometry, not data: build them once, outside the loop.
    s_coords = _make_s_coords(nx, ny, device, torch.float32)
    flat_channel_index(nx, ny, nz, device)
    nx * ny

    # One fixed validation subsample, drawn once from a seeded generator so the
    # val loss is comparable across epochs and across runs.
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    loss_history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    output_dir = Path("data/backward_model/training_logs")
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    patience = 80
    epochs_without_improvement = 0

    print(
        f"Training on {len(train_dataset)} samples, "
        f"validating on {len(val_dataset)} samples..."
    )

    for epoch in range(epochs):
        # --- TRAIN ---
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch")

        for batch_idx, (params_batch, s_mat_batch) in enumerate(pbar):
            params_batch = params_batch.to(device, dtype=torch.float32)  # noqa: PLW2901
            s_mat_batch = s_mat_batch.to(device, dtype=torch.float32)  # noqa: PLW2901

            b = s_mat_batch.shape[0]
            # |S|^2 already -- the solver returns abs-squared, so no squaring here.
            s_flat = s_mat_batch.reshape(b, -1)

            optimizer.zero_grad()
            pred = model(s_coords, s_flat)
            loss = criterion(pred, params_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix(
                loss=f"{loss.item():.3e}",
                avg=f"{train_loss / (batch_idx + 1):.3e}",
                lr=f"{optimizer.param_groups[0]['lr']:.1e}",
            )

        # --- VALIDATE ---
        model.eval()
        val_loss = 0.0
        pbar_val = tqdm(val_loader, desc=f"Val {epoch + 1}/{epochs}", unit="batch")

        with torch.no_grad():
            for batch_idx, (params_batch, s_mat_batch) in enumerate(pbar_val):
                params_batch = params_batch.to(device, dtype=torch.float32)  # noqa: PLW2901
                s_mat_batch = s_mat_batch.to(device, dtype=torch.float32)  # noqa: PLW2901

                b = s_mat_batch.shape[0]
                pred = model(s_coords, s_mat_batch.reshape(b, -1))
                delta_loss = criterion(pred, params_batch).item()

                val_loss += delta_loss
                pbar_val.set_postfix(
                    loss=f"{delta_loss:.3e}", avg=f"{val_loss / (batch_idx + 1):.3e}"
                )

        average_train_loss = train_loss / len(train_loader)
        average_val_loss = val_loss / len(val_loader)
        loss_history["train_loss"].append(average_train_loss)
        loss_history["val_loss"].append(average_val_loss)
        save_loss_history(loss_history, output_dir / "loss_history.json")

        scheduler.step(average_val_loss)
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1:03d}/{epochs} | "
            f"Loss (Tr/Val): {average_train_loss:.1e} / {average_val_loss:.1e} "
            f"[LR: {lr:.1e}]"
        )

        if average_val_loss < best_val_loss:
            best_val_loss = average_val_loss
            torch.save(
                model.state_dict(), "data/backward_model/best_backward_model-2.pth"
            )
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print("Early stopping triggered!")
            break

    print("Training complete.")
    save_loss_history(loss_history, output_dir / "loss_history.json")
    print(f"--> Saved metrics data to: {output_dir / 'loss_history.json'}")
    plot_training_convergence(loss_history, output_dir / "convergence_curve.png")
    torch.save(model.state_dict(), "data/backward_model/backward_model-2.pth")


OUTPUT_DIR = Path("data/backward_model")


PARAM_LABELS = (
    (r"$D$", r"$\mathrm{meV}$"),
    (r"$h$", r"$\mathrm{\AA}$"),
    (r"$\beta$", ""),
    (r"$a$", r"$\mathrm{\AA}$"),
    (r"$\theta$", r"$\circ$"),
)


def _denormalize_params(params_norm: np.ndarray) -> np.ndarray:
    """(N, 5) in [0, 1] -> physical units."""
    lo = np.asarray(PARAMS_MIN, dtype=np.float64)
    hi = np.asarray(PARAMS_MAX, dtype=np.float64)
    return lo + params_norm * (hi - lo)


def test(  # ruff: ignore[too-many-arguments]  # ruff: ignore[too-many-locals]
    n_samples: int = 200,  # ruff: ignore[pytest-parameter-with-default-argument]
    nx: int = 11,  # ruff: ignore[pytest-parameter-with-default-argument]
    ny: int = 11,  # ruff: ignore[pytest-parameter-with-default-argument]
    *,
    checkpoint: str = "best_backward_model-2.pth",  # ruff: ignore[pytest-parameter-with-default-argument]
    seed: int = 0,  # ruff: ignore[pytest-parameter-with-default-argument]
    save: bool = True,  # ruff: ignore[pytest-parameter-with-default-argument]
) -> None:
    """Recover the 5 potential parameters from many simulated S matrices."""
    backward_model = JointKernelBackwardNet().to(DEVICE)
    backward_model.load_state_dict(
        torch.load(
            OUTPUT_DIR / checkpoint,
            map_location=DEVICE,
        ),
    )
    backward_model.eval()

    rng = np.random.default_rng(seed)
    true_norm = rng.uniform(size=(n_samples, 5))

    s_list = [
        np.asarray(simulate_s_matrix(p), dtype=np.float32)
        for p in tqdm(true_norm, desc="simulating", unit="sample")
    ]
    s_matrix = torch.as_tensor(np.stack(s_list, axis=0), device=DEVICE)  # (N, Nx, Ny)

    pred_norm = (
        predict_potential_params(
            model=backward_model,
            s_matrix=s_matrix,
            nx=nx,
            ny=ny,
        )
        .cpu()
        .numpy()
    )  # (N, 5)

    true_phys = _denormalize_params(true_norm)
    pred_phys = _denormalize_params(pred_norm.astype(np.float64))

    lo = np.asarray(PARAMS_MIN, dtype=np.float64)
    hi = np.asarray(PARAMS_MAX, dtype=np.float64)
    span = hi - lo

    err = pred_phys - true_phys
    rmse = np.sqrt(np.mean(err**2, axis=0))
    mae = np.mean(np.abs(err), axis=0)
    bias = np.mean(err, axis=0)
    ss_res = np.sum(err**2, axis=0)
    ss_tot = np.sum((true_phys - true_phys.mean(axis=0)) ** 2, axis=0)
    r2 = 1.0 - ss_res / ss_tot
    pearson = np.array(
        [np.corrcoef(true_phys[:, k], pred_phys[:, k])[0, 1] for k in range(5)]
    )

    names = [lab for lab, _ in PARAM_LABELS]
    print(f"\nparameter recovery, n = {n_samples}")
    print(
        f"{'param':>8} {'R^2':>8} {'r':>8} {'RMSE':>10} "
        f"{'RMSE/range':>11} {'bias':>10} {'MAE':>10}"
    )
    for k, name in enumerate(names):
        print(
            f"{name.strip('$'):>8} {r2[k]:8.4f} {pearson[k]:8.4f} {rmse[k]:10.4f} "
            f"{100 * rmse[k] / span[k]:10.2f}% {bias[k]:10.4f} {mae[k]:10.4f}"
        )

    # --- scatter: predicted vs true, one panel per parameter ---
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.2))

    for k, ax in enumerate(axes):
        label, unit = PARAM_LABELS[k]
        ax.scatter(true_phys[:, k], pred_phys[:, k], s=12, alpha=0.5, edgecolors="none")
        ax.plot([lo[k], hi[k]], [lo[k], hi[k]], color="k", lw=1, ls="--")
        ax.set_xlim(lo[k], hi[k])
        ax.set_ylim(lo[k], hi[k])
        ax.set_aspect("equal")
        axis_label = f"{label}" + (f"  [{unit}]" if unit else "")
        ax.set_xlabel(f"true {axis_label}")
        ax.set_ylabel(f"predicted {axis_label}")
        ax.set_title(
            f"$R^2$ {r2[k]:.4f}, RMSE {100 * rmse[k] / span[k]:.1f}% of range",
            fontsize=10,
        )

    fig.suptitle(f"parameter recovery from $|S|^2$ (n = {n_samples})", fontsize=12)
    fig.tight_layout()
    if save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUTPUT_DIR / "backward_params.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    RUN_GENERATE = False
    RUN_TRAIN = False
    RUN_TEST = False

    if RUN_GENERATE:
        generate()
    if RUN_TRAIN:
        train()
    if RUN_TEST:
        test()
