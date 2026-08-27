import contextlib
import json
import time
from pathlib import Path
from typing import Any, override

import h5py  # type: ignore[import-untyped]
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from multiscat import OptimizationConfig, get_scattering_matrix
from multiscat.basis import (
    scattering_metadata_from_stacked_delta_x,
    split_scattering_metadata,
)
from multiscat.config import MorseScatteringCondition, momentum_from_angles
from scipy.constants import angstrom as angstrom_si  # type: ignore[import-untyped]
from scipy.constants import (  # type: ignore[import-untyped]
    electron_volt,
    physical_constants,
)
from slate_core import Array, AsUpcast, basis, plot
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

PARAMS_MIN = np.array([5.0, 0.5, 0.5, 0.05, 0.5, 0.0], dtype=np.float64)
PARAMS_MAX = np.array([10.0, 1.5, 4.0, 0.20, 4.0, np.pi / 2], dtype=np.float64)
Nx, Ny, Nz = 15, 15, 200
potential_channels = Nx
potential_z = 100
z = torch.linspace(0.0, Z_HEIGHT, potential_z)


def denormalize_params(
    params_norm: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    """Scales parameters back to their original physical units."""
    return params_norm * (PARAMS_MAX - PARAMS_MIN) + PARAMS_MIN


def condition_from_params(
    params: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> MorseScatteringCondition:
    """Convert a tensor of parameters into a ScatteringCondition."""
    depth, height, offset, beta, unit_cell, theta = denormalize_params(
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
            np.array([unit_cell * angstrom_si, 0, 0]),
            np.array(
                [
                    0,
                    unit_cell * angstrom_si,
                    0,
                ]
            ),
            np.array([0, 0, Z_HEIGHT * angstrom_si]),
        ),
        (Nx, Ny, Nz),
    )

    return MorseScatteringCondition(
        mass=HELIUM_MASS,
        morse_parameters=morse_params,
        metadata=metadata,
        incident_k=momentum_from_angles(
            theta=theta,
            phi=np.deg2rad(0),
            energy=HELIUM_ENERGY,
            mass=HELIUM_MASS,
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
    metadata_x01, _ = split_scattering_metadata(condition.metadata)
    unit_cell = metadata_x01.children[0].delta / angstrom_si

    morse_parameters = condition.morse_parameters

    depth = morse_parameters.depth / (electron_volt * 10**-3)
    height = morse_parameters.height / angstrom_si
    offset = morse_parameters.offset / angstrom_si
    beta = morse_parameters.beta

    theta = condition.theta

    return normalize_params(np.array([depth, height, offset, beta, unit_cell, theta]))


def simulate_s_matrix(
    params: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    """Wrap your physics code into a single callable function."""
    condition = condition_from_params(params)

    config = OptimizationConfig(precision=1e-5, max_iterations=1000, n_channels=80)
    s_matrix = get_scattering_matrix(condition, config, backend="scipy")

    metadata_x01, _ = split_scattering_metadata(condition.metadata)
    return s_matrix.with_basis(
        AsUpcast(basis.transformed_from_metadata(metadata_x01), metadata_x01),
    ).raw_data.real.reshape(15, 15)


def corrugated_morse_channels(  # ruff: ignore[too-many-arguments]
    z: torch.Tensor | np.ndarray,
    *,
    depth: float,
    height: float,
    offset: float,
    beta: float,
    n_channels: int = potential_channels,
) -> torch.Tensor:
    """Return the reciprocal space potential in the shape (n_channels, n_channels, nz)."""
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
    """Return the real space potential in the shape (n_channels, n_channels, nz)."""
    z = torch.as_tensor(z, device=DEVICE, dtype=torch.float32)
    x = torch.as_tensor(x, device=DEVICE, dtype=torch.float32)
    y = torch.as_tensor(y, device=DEVICE, dtype=torch.float32)

    t = torch.exp(-(z - offset) / height)
    v0 = depth * (t**2 - 2 * t)
    v1 = -2 * beta * depth * t**2

    q = torch.cos(2 * np.pi * x / a) + torch.cos(2 * np.pi * y / a)

    return v0 + (v1 * q)


def get_potential(params: torch.Tensor | np.ndarray, z: torch.Tensor) -> torch.Tensor:
    if isinstance(params, torch.Tensor):
        params = params.detach().cpu().numpy()
    depth, height, offset, beta, *_ = denormalize_params(
        np.asarray(params, dtype=np.float64)
    )
    return corrugated_morse_channels(
        z,
        depth=float(depth),
        height=float(height),
        offset=float(offset),
        beta=float(beta),
    )


class GlobalPotentialEncoder(nn.Module):
    """
    Learn the mapping from the potential coordinates (n, m, z) to a vector of size out_dim.

    The out_dim depends on the complexity of the potential. This map is used to build the encoder to be fed to the core model.
    """

    def __init__(self, grid_shape: tuple[int, int, int], out_dim: int) -> None:
        super().__init__()
        self.grid_shape = tuple(grid_shape)
        self.Npts = int(np.prod(self.grid_shape))
        self.net = nn.Sequential(
            nn.Linear(self.Npts, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, out_dim),
        )

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, Nx, Ny, Nz)
        if x.ndim == 5:  # ruff: ignore[magic-value-comparison]
            x = x[:, 0]  # -> (B, Nx, Ny, Nz), non-contiguous
        elif x.ndim != 4:  # ruff: ignore[magic-value-comparison]
            msg = (
                f"expected (B, 2, Nx, Ny, Nz) or (B, Nx, Ny, Nz), got {tuple(x.shape)}"
            )
            raise ValueError(msg)

        if x.shape[1:] != self.grid_shape:
            msg = f"grid mismatch: got {tuple(x.shape[1:])}, expected {self.grid_shape}"
            raise ValueError(msg)

        return self.net(x.reshape(x.shape[0], self.Npts))


class ResBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout_rate: float = 0.05) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),  # Added Dropout
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout_rate),  # Added Dropout
        )
        self.act = nn.GELU()

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The skip connection: add the original input to the transformed output
        return self.act(x + self.net(x))


class ForwardModel(nn.Module):
    """Predicts S (one positive scalar per (n,m) point) from particle params and the local and the global features of the reciprocal-space potential lattice."""

    def __init__(  # ruff: ignore[too-many-arguments]  # ruff: ignore[too-many-positional-arguments]
        self,
        grid_shape: tuple[int, int, int] = (
            1,
            1,
            potential_z,
        ),
        input_dim: int = 2,
        coord_dim: int = 2,
        encoder_dim: int = 64,
        hidden_dim: int = 128,
        output_shape: int = 1,
    ) -> None:
        super().__init__()

        self.grid_shape = grid_shape
        self.output_shape = output_shape
        self.Npts = int(np.prod(grid_shape))
        # learnt global feature from a sub-network, increase the encoder_dim if the potential gets more complicated
        self.global_encoder = GlobalPotentialEncoder((Nx, Ny, potential_z), encoder_dim)

        in_dim = input_dim + self.Npts + encoder_dim + coord_dim
        self.embedding = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.res_blocks = nn.Sequential(*(ResBlock(hidden_dim) for _ in range(4)))

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, int(np.prod(output_shape))),
            nn.LeakyReLU(negative_slope=0.01),
        )

    @override
    def forward(
        self,
        coords: torch.Tensor,  # (B*N, 2)
        params: torch.Tensor,  # (B*N, input_dim)
        local_potential: torch.Tensor,  # (B*N, Nz)
        global_feature: torch.Tensor,  # (B*N, encoder_dim)
    ) -> torch.Tensor:
        # mix and align the inputs to match the shape of coords for a pointwise mapping
        x = self.embedding(
            torch.cat([params, coords, local_potential, global_feature], dim=-1)
        )
        x = self.res_blocks(x)
        out = self.head(x)
        return out.view(-1)


def _make_coords(
    nx: int,
    ny: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    xs = torch.fft.fftfreq(nx, d=1.0 / nx).to(device=device, dtype=dtype)
    ys = torch.fft.fftfreq(ny, d=1.0 / ny).to(device=device, dtype=dtype)

    # Create the 2D grid
    grid = torch.stack(
        torch.meshgrid(xs, ys, indexing="ij"),
        dim=-1,
    )  # (Nx, Ny, 2)

    return grid.reshape(-1, 2)  # (N, 2)


PARAM_IDX = [-2, -1]  # unit_cell, theta


def predict_chi_batch_from_params(  # ruff: ignore[too-many-arguments]  # ruff: ignore[too-many-positional-arguments]
    forward_model: nn.Module,
    params_batch: torch.Tensor,
    coords: torch.Tensor,
    nx: int,
    ny: int,
    z: torch.Tensor,
) -> torch.Tensor:
    """Format the model ouput for comparison with the target."""
    b = params_batch.shape[0]
    n_pts = coords.shape[0]
    device, dtype = params_batch.device, params_batch.dtype

    coords = coords.to(device=device, dtype=dtype)
    potential_batch = torch.stack(
        [get_potential(p, z) for p in params_batch], dim=0
    ).to(device=device, dtype=dtype)

    # per-condition quantities: computed once, then repeated N times each
    global_feature = forward_model.global_encoder(potential_batch)  # ty: ignore[call-non-callable]
    # feed the original potential along with the learnt global feature
    local = potential_batch[:, coords[:, 0].long(), coords[:, 1].long()]  # (B, N, Nz)

    coords_flat = coords.unsqueeze(0).expand(b, n_pts, 2).reshape(b * n_pts, 2)
    params_flat = params_batch[:, PARAM_IDX].repeat_interleave(n_pts, dim=0)
    global_flat = global_feature.repeat_interleave(n_pts, dim=0)
    local_flat = local.reshape(b * n_pts, -1)

    return forward_model(
        coords=coords_flat,
        params=params_flat,
        local_potential=local_flat,
        global_feature=global_flat,
    ).view(b, nx, ny)


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
        input_data = f.create_dataset("X", shape=(num_samples, 6), dtype=np.float64)  # type: ignore[hd5]
        output_data = f.create_dataset(  # type: ignore[hd5]
            "Y",
            shape=(num_samples, 15, 15),
            dtype=np.float64,
        )

        for i in range(num_samples):
            print(f"Generating sample {i + 1}/{num_samples}")
            params = rng.uniform(size=6)

            input_data[i] = params
            output_data[i] = simulate_s_matrix(params)


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
        y_tensor = torch.tensor(self.file["Y"][idx]).float()  # type: ignore[untyped]

        return x_tensor, y_tensor

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
    for i in range(50):
        data_path = Path(f"data/15/scattering_pointwise/training_data/dataset.{i}.hdf5")
        generate_dataset_hdf5(data_path, num_samples=1000)


def load_datasets() -> ConcatDataset[tuple[torch.Tensor, torch.Tensor]]:
    datasets = [
        HDF5ScatteringDataset(
            Path(f"data/15/scattering_pointwise/training_data/dataset.{i}.hdf5")
        )
        for i in range(50)
    ]
    return ConcatDataset[tuple[torch.Tensor, torch.Tensor]](datasets)


class SparseScatteringLoss(nn.Module):
    def __init__(
        self,
        peak_weight: float = 10.0,
        sparsity_weight: float = 1e-4,
    ) -> None:
        super().__init__()
        self.peak_weight = peak_weight
        self.sparsity_weight = sparsity_weight
        self.mse = nn.MSELoss(reduction="none")  # Notice reduction='none'

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # 1. Calculate the raw, un-averaged pixel-wise squared error
        base_error = self.mse(y_pred, y_true)

        # 2. Intensity Weighting
        # Create a mask where empty channels equal 1.0, and bright channels > 1.0
        # Example: If a peak has intensity 0.1 and weight is 100, its multiplier becomes 11.0
        weight_mask = 1.0 + (self.peak_weight * y_true)

        # Apply the weight and take the mean
        weighted_mse = torch.mean(base_error * weight_mask)

        # 3. Sparsity Penalty (L1)
        # This constantly applies a tiny downward pressure on all predicted values,
        # forcing the network to snap the background noise to exactly 0.0
        sparsity_loss = torch.mean(torch.abs(y_pred))

        return weighted_mse + (self.sparsity_weight * sparsity_loss)


def train() -> None:  # noqa: PLR0914   # ruff: ignore[too-many-statements]
    dataset = load_datasets()
    train_dataset, val_dataset = random_split(dataset, [0.8, 0.2])

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    coords = _make_coords(Nx, Ny, device=DEVICE, dtype=torch.float32)

    # 2. Initialize Models
    forward_model = ForwardModel().to(DEVICE)
    # maybe use TotalScatteringLoss(lambda_physics=0.01)
    forward_criterion = SparseScatteringLoss(peak_weight=10.0, sparsity_weight=1e-5)
    forward_optimizer = optim.AdamW(
        forward_model.parameters(),
        lr=1e-3,
        weight_decay=1e-5,
    )

    scheduler_f = optim.lr_scheduler.ReduceLROnPlateau(
        forward_optimizer,
        mode="min",
        factor=0.5,
        patience=5,
    )
    loss_history = {"train_loss": [], "val_loss": []}
    output_dir = Path("data/full_potential_model_demo/training_logs")
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss_f = float("inf")
    patience = 80
    epochs_without_improvement = 0
    epochs = 2000
    print(
        f"Training on {len(train_dataset)} samples,"
        f"Validating on {len(val_dataset)} samples...",
    )
    print(f"Using device: {DEVICE}")

    for epoch in range(epochs):
        forward_model.train()

        train_loss_f = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch")

        for batch_idx, (
            params_batch,
            s_mat_batch,
        ) in enumerate(pbar):
            t0 = time.perf_counter()
            params_batch = params_batch.to(DEVICE)  # noqa: PLW2901
            s_mat_batch = s_mat_batch.to(DEVICE)  # noqa: PLW2901
            # --- Forward Model Update ---
            forward_optimizer.zero_grad()
            s_mat_pred = predict_chi_batch_from_params(
                forward_model, params_batch, coords, Nx, Ny, z
            )
            t1 = time.perf_counter()
            loss_f = forward_criterion(s_mat_pred, s_mat_batch)
            loss_f.backward()
            t2 = time.perf_counter()
            forward_optimizer.step()
            t3 = time.perf_counter()
            train_loss_f += loss_f.item()
            avg_loss = train_loss_f / (batch_idx + 1)
            pbar.set_postfix(
                loss=f"{loss_f.item():.3e}",
                avg=f"{avg_loss:.3e}",
                pred=f"{t1 - t0:.2f}s",
                back=f"{t2 - t1:.2f}s",
                step=f"{t3 - t2:.2f}s",
                lr=f"{forward_optimizer.param_groups[0]['lr']:.1e}",
            )

        # --- VALIDATION PHASE ---
        forward_model.eval()

        val_loss_f = 0.0
        pbar_val = tqdm(val_loader, desc=f"Val {epoch + 1}/{epochs}", unit="batch")

        with torch.no_grad():
            for batch_idx, (
                params_batch,
                s_mat_batch,
            ) in enumerate(pbar_val):
                params_batch = params_batch.to(DEVICE)  # noqa: PLW2901
                s_mat_batch = s_mat_batch.to(DEVICE)  # noqa: PLW2901

                # Forward Model Validation
                s_mat_pred = predict_chi_batch_from_params(
                    forward_model, params_batch, coords, Nx, Ny, z
                )
                delta_loss = forward_criterion(s_mat_pred, s_mat_batch).item()
                val_loss_f += delta_loss
                avg_val = val_loss_f / (batch_idx + 1)
                pbar_val.set_postfix(
                    loss=f"{delta_loss:.3e}",
                    avg=f"{avg_val:.3e}",
                )

        # Averages
        average_train_loss_f = train_loss_f / len(train_loader)
        average_val_loss_f = val_loss_f / len(val_loader)
        loss_history["train_loss"].append(average_train_loss_f)
        loss_history["val_loss"].append(average_val_loss_f)
        save_loss_history(loss_history, output_dir / "loss_history.json")

        # Step the schedulers
        scheduler_f.step(average_val_loss_f)

        # Retrieve current learning rates for logging
        lr_f = forward_optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch + 1:03d}/{epochs} | "
            f"Fwd Loss (Tr/Val): {average_train_loss_f:.1e} / {average_val_loss_f:.1e}"
            f"[LR: {lr_f:.1e}] | "
        )

        if average_val_loss_f < best_val_loss_f:
            best_val_loss_f = average_val_loss_f
            torch.save(
                forward_model.state_dict(),
                "data/full_potential_model_demo/best_forward_model_potential_full.pth",
            )
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print("Early stopping triggered!")
            break

    print("Training complete.")

    with Path(output_dir / "loss_history.json").open("w", encoding="utf-8") as f:
        json.dump(loss_history, f, indent=4)
    print(f"--> Saved metrics data to: {output_dir / 'loss_history.json'}")

    plot_training_convergence(loss_history, output_dir / "convergence_curve.png")

    torch.save(
        forward_model.state_dict(),
        "data/full_potential_model_demo/forward_model_potential_full.pth",
    )


OUTPUT_DIR = Path("data/full_potential_model_demo")


def compute_metrics(
    predicted: np.ndarray[Any, Any],
    actual: np.ndarray[Any, Any],
    name: str,
    *,
    verbose: bool = True,
) -> dict[str, float | str]:
    """Regression metrics between predicted and exact arrays.

    Returns the metrics as a dict (so they can be put in a plot title) as well
    as optionally printing them.
    """
    predicted_np = np.asarray(predicted)
    actual_np = np.asarray(actual)

    difference = predicted_np - actual_np

    mse = float(np.mean(np.abs(difference) ** 2))
    rmse = float(np.sqrt(mse))

    actual_norm = float(np.linalg.norm(actual_np.ravel()))
    predicted_norm = float(np.linalg.norm(predicted_np.ravel()))

    relative_l2 = float(np.linalg.norm(difference.ravel()) / (actual_norm + 1e-12))

    if np.iscomplexobj(predicted_np) or np.iscomplexobj(actual_np):
        correlation = float(
            np.abs(np.vdot(actual_np.ravel(), predicted_np.ravel()))
            / (actual_norm * predicted_norm + 1e-12)
        )
        correlation_name = "CosSim"
    else:
        actual_std = float(np.std(actual_np))
        predicted_std = float(np.std(predicted_np))
        if actual_std < 1e-12 or predicted_std < 1e-12:  # ruff: ignore[magic-value-comparison]
            correlation = float("nan")
        else:
            correlation = float(
                np.corrcoef(actual_np.ravel(), predicted_np.ravel())[0, 1]
            )
        correlation_name = "Corr"

    residual_sum = float(np.sum(np.abs(difference) ** 2))
    total_sum = float(np.sum(np.abs(actual_np - np.mean(actual_np)) ** 2) + 1e-12)
    r_squared = float(1.0 - residual_sum / total_sum)

    if verbose:
        print(f"\n[{name}]")
        print(f"MSE    : {mse:.6e}")
        print(f"RMSE   : {rmse:.6e}")
        print(f"RelL2  : {relative_l2:.6e}")
        print(f"{correlation_name:<7}: {correlation:.6f}")
        print(f"R^2    : {r_squared:.6f}")

    return {
        "mse": mse,
        "rmse": rmse,
        "rel_l2": relative_l2,
        "corr": correlation,
        "r2": r_squared,
        "corr_name": correlation_name,  # type: ignore[dict-item]
    }


def _draw_panel(
    array: Array[Any, np.dtype[np.complexfloating]],
    ax: Axes,
    title: str,
) -> PolyCollection:
    """Draw one slate array into a supplied Axes and strip its built-in colour bar."""
    _fig, _ax, mesh = plot.array_against_axes_2d_k_nearest_neighbor(
        array,
        ax=ax,
        measure="abs",
    )
    # The slate helper attaches its own colour bar to the mesh. Remove it so a
    # single shared bar can be added for the whole figure.
    colorbar = getattr(mesh, "colorbar", None)
    if colorbar is not None:
        colorbar.remove()
        mesh.colorbar = None
    ax.set_title(title, fontsize=11)
    return mesh


def test() -> None:  # ruff: ignore[too-many-locals]
    """Evaluate the potential-encoder surrogate on a fixed test condition."""
    condition = MorseScatteringCondition(
        mass=HELIUM_MASS,
        morse_parameters=operator.build.CorrugatedMorseParameters(
            depth=7.63 * electron_volt * 10**-3,
            height=(1.0 / 1.1) * angstrom_si,
            offset=1.0 * angstrom_si,
            beta=0.05,
        ),
        metadata=scattering_metadata_from_stacked_delta_x(
            (
                np.array([3.0 * angstrom_si, 0, 0]),
                np.array([0, 3.0 * angstrom_si, 0]),
                np.array([0, 0, Z_HEIGHT * angstrom_si]),
            ),
            (15, 15, 200),
        ),
        incident_k=momentum_from_angles(
            theta=np.deg2rad(45),
            phi=np.deg2rad(0),
            energy=HELIUM_ENERGY,
            mass=HELIUM_MASS,
        ),
    )

    test_params = torch.tensor(
        params_from_condition(condition),
        dtype=torch.float32,
    ).unsqueeze(0)
    nx, ny, _nz = condition.metadata.shape

    forward_model = ForwardModel().to(DEVICE)
    forward_model.load_state_dict(
        torch.load(
            OUTPUT_DIR / "best_forward_model_potential_full.pth",
            map_location=DEVICE,
        ),
    )
    coords = _make_coords(nx, ny, device=DEVICE, dtype=torch.float32)
    forward_model.eval()

    with torch.no_grad():
        channel_intensity_dense = predict_chi_batch_from_params(
            forward_model,
            test_params.to(DEVICE),
            coords,
            nx,
            ny,
            z,
        )[0]
        metadata_x01, _ = split_scattering_metadata(condition.metadata)
        predict = Array(
            AsUpcast(basis.transformed_from_metadata(metadata_x01), metadata_x01),
            channel_intensity_dense.detach().cpu().numpy().astype(np.complex128),
        )

    actual = get_scattering_matrix(
        condition,
        OptimizationConfig(precision=1e-5, max_iterations=1000, n_channels=400),
        backend="scipy",
    )

    error = actual - predict

    # ------------------------------------------------------------------
    # Metrics first, so they can be written into the figure title
    # ------------------------------------------------------------------
    metrics = compute_metrics(
        predict.raw_data.real,
        actual.raw_data.real,
        name="Scattering matrix (encoder model)",
    )

    print(metrics)

    # ------------------------------------------------------------------
    # Combined figure: predicted | exact | error, one shared colour bar
    # ------------------------------------------------------------------
    panels: list[tuple[Any, str]] = [
        (predict, "Prediction"),
        (actual, "Actual"),
        (error, "Error  |exact - predicted|"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True)
    meshes = [
        _draw_panel(array, ax, title)
        for ax, (array, title) in zip(axes, panels, strict=True)
    ]

    # Shared colour scale across all three panels.
    all_values = np.concatenate(
        [np.abs(np.asarray(array.raw_data)).ravel() for array, _ in panels]
    )
    vmin, vmax = float(all_values.min()), float(all_values.max())
    for mesh in meshes:
        mesh.set_clim(vmin, vmax)

    cbar = fig.colorbar(
        meshes[0],
        ax=list(axes),
        shrink=0.9,
        pad=0.02,
        aspect=30,
    )
    cbar.set_label(r"$|S|$", fontsize=11)

    corr_name = metrics["corr_name"]
    fig.suptitle(
        "S-matrix "
        r"($\theta = 45^\circ$, $\phi = 0^\circ$, "
        r"$D = 7.63$ meV, $a = 3$ Å, "
        rf"number of channels = {nx * ny})"
        "\n"
        f"RelL2 = {metrics['rel_l2']:.3e}    "
        f"{corr_name} = {metrics['corr']:.4f}    "
        f"$R^2$ = {metrics['r2']:.4f}    "
        f"RMSE = {metrics['rmse']:.3e}    "
        f"$\\Sigma|\\Delta| $ = {np.sum(np.abs(error.raw_data)):.3e}",
        fontsize=20,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        OUTPUT_DIR / f"scattering_comparison_full_{nx * ny}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)
    print(f"--> Saved {OUTPUT_DIR / f'scattering_comparison_full_{nx * ny}.png'}")


if __name__ == "__main__":
    RUN_GENERATE = False
    RUN_TRAIN = True
    RUN_TEST = True

    if RUN_GENERATE:
        generate()
    if RUN_TRAIN:
        train()
    if RUN_TEST:
        test()


# Grids to evaluate. The model was trained on 15x15 only.
GRID_SIZES = (10, 11, 12, 13, 14, 15, 18, 20, 22, 24, 25)
TRAINED_GRID = 15

# Only these get a printed value -- annotating every point is unreadable.
ANNOTATED_GRIDS = (10, 15, 25)

# Type sizes, tuned for projection rather than for a page.
FONT = {
    "tick": 15,
    "axis_label": 17,
    "panel_title": 18,
    "annotation": 16,
    "suptitle": 19,
}

# Ground-truth solver settings.
GROUND_TRUTH_CHANNELS = 400


def _condition_for_grid(n_side: int) -> MorseScatteringCondition:
    """Build the fixed test condition on an ``n_side x n_side`` channel grid."""
    return MorseScatteringCondition(
        mass=HELIUM_MASS,
        morse_parameters=operator.build.CorrugatedMorseParameters(
            depth=7.63 * electron_volt * 10**-3,
            height=(1.0 / 1.1) * angstrom_si,
            offset=1.0 * angstrom_si,
            beta=0.05,
        ),
        metadata=scattering_metadata_from_stacked_delta_x(
            (
                np.array([3.0 * angstrom_si, 0, 0]),
                np.array([0, 3.0 * angstrom_si, 0]),
                np.array([0, 0, Z_HEIGHT * angstrom_si]),
            ),
            (n_side, n_side, 200),
        ),
        incident_k=momentum_from_angles(
            theta=np.deg2rad(45),
            phi=np.deg2rad(0),
            energy=HELIUM_ENERGY,
            mass=HELIUM_MASS,
        ),
    )


def _evaluate_grid(
    forward_model: ForwardModel,
    n_side: int,
) -> tuple[Any, Any, dict[str, float | str]]:
    """Run the surrogate and the exact solver on one grid size."""
    condition = _condition_for_grid(n_side)
    nx, ny, _nz = condition.metadata.shape

    test_params = torch.tensor(
        params_from_condition(condition),
        dtype=torch.float32,
    ).unsqueeze(0)
    coords = _make_coords(nx, ny, device=DEVICE, dtype=torch.float32)

    with torch.no_grad():
        channel_intensity_dense = predict_chi_batch_from_params(
            forward_model,
            test_params.to(DEVICE),
            coords,
            nx,
            ny,
            z,
        )[0]
        metadata_x01, _ = split_scattering_metadata(condition.metadata)
        predict = Array(
            AsUpcast(basis.transformed_from_metadata(metadata_x01), metadata_x01),
            channel_intensity_dense.detach().cpu().numpy().astype(np.complex128),
        )

    actual = get_scattering_matrix(
        condition,
        OptimizationConfig(
            precision=1e-5,
            max_iterations=1000,
            n_channels=GROUND_TRUTH_CHANNELS,
        ),
        backend="scipy",
    )

    metrics = compute_metrics(
        predict.raw_data.real,
        actual.raw_data.real,
        name=f"Scattering matrix ({nx * ny} channels)",
    )
    return predict, actual, metrics


COMPARISON_SWEEPS = {
    Path(OUTPUT_DIR / "sweep_fixed.npz"): "fixed encoder",
    Path("data/full_potential_model_flex/2/sweep_mesh_free.npz"): "mesh-free encoder",
}


def test_2() -> None:  # ruff: ignore[too-many-locals] #ruff: ignore[too-many-statements]
    """Evaluate the surrogate across grid sizes and build the slide figure."""
    plt.rcParams.update(
        {
            "font.size": FONT["tick"],
            "axes.labelsize": FONT["axis_label"],
            "axes.titlesize": FONT["panel_title"],
            "xtick.labelsize": FONT["tick"],
            "ytick.labelsize": FONT["tick"],
        }
    )

    forward_model = ForwardModel().to(DEVICE)
    forward_model.load_state_dict(
        torch.load(
            "data/full_potential_model/best_forward_model_potential_full.pth",
            map_location=DEVICE,
        ),
    )
    forward_model.eval()

    results = {n: _evaluate_grid(forward_model, n) for n in GRID_SIZES}

    np.array([n * n for n in GRID_SIZES], dtype=float)
    rel_l2 = np.array([float(results[n][2]["rel_l2"]) for n in GRID_SIZES])
    np.savez(
        OUTPUT_DIR / "sweep_fixed.npz",
        grids=np.array(GRID_SIZES),
        rel_l2=rel_l2,
    )

    # ------------------------------------------------------------------
    # Figure: two image panels (largest grid) + the RelL2 trend
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(15, 5.0), constrained_layout=True)
    gridspec = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.5])
    ax_pred = fig.add_subplot(gridspec[0, 0])
    ax_exact = fig.add_subplot(gridspec[0, 1])
    ax_trend = fig.add_subplot(gridspec[0, 2])

    show_grid = max(GRID_SIZES)
    predict, actual, _ = results[show_grid]

    mesh_pred = _draw_panel(predict, ax_pred, "Prediction")
    mesh_exact = _draw_panel(actual, ax_exact, "Exact")

    # _draw_panel hard-codes fontsize=11 on the title; override it here.
    for ax in (ax_pred, ax_exact):
        ax.title.set_fontsize(FONT["panel_title"])

    shared = np.concatenate(
        [
            np.abs(np.asarray(predict.raw_data)).ravel(),
            np.abs(np.asarray(actual.raw_data)).ravel(),
        ]
    )
    vmin, vmax = float(shared.min()), float(shared.max())
    for mesh in (mesh_pred, mesh_exact):
        mesh.set_clim(vmin, vmax)

    # Reciprocal-space axes in inverse angstroms rather than 1e11 m^-1.
    for ax in (ax_pred, ax_exact):
        ax.ticklabel_format(style="plain", axis="both")
        ax.set_xlabel(r"$k_0$ / $\mathrm{\AA}^{-1}$")
        ax.set_ylabel("")
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_formatter(
                plt.FuncFormatter(lambda v, _pos: f"{v * 1e-10:.0f}")
            )
    ax_pred.set_ylabel(r"$k_1$ / $\mathrm{\AA}^{-1}$")

    cbar = fig.colorbar(
        mesh_pred,
        ax=[ax_pred, ax_exact],
        shrink=0.85,
        pad=0.02,
        aspect=28,
    )
    cbar.ax.tick_params(labelsize=FONT["tick"])

    # ------------------------------------------------------------------
    # Panel B: RelL2 vs channel count, fixed y-range so flat looks flat
    # ------------------------------------------------------------------
    curves = []
    for path, label in COMPARISON_SWEEPS.items():
        d = np.load(path)
        curves.append((label, d["grids"], d["rel_l2"]))

    styles = (
        {"color": "#1f4e79", "marker": "o", "linestyle": "-"},
        {"color": "#c1440e", "marker": "s", "linestyle": "--"},
    )
    label_offsets = (18, -30)  # first curve above its marker, second below
    for (label, grids, values), style, dy in zip(
        curves, styles, label_offsets, strict=False
    ):
        ax_trend.semilogx(
            grids**2,
            values,
            markersize=9,
            linewidth=2,
            label=label,
            zorder=3,
            **style,
        )
        for n_side, value in zip(grids, values, strict=True):
            if int(n_side) not in ANNOTATED_GRIDS:
                continue
            ax_trend.annotate(
                f"{value:.3f}",
                xy=(n_side**2, value),
                xytext=(0, dy),
                textcoords="offset points",
                ha="center",
                fontsize=FONT["annotation"],
                color=style["color"],
                fontweight="bold",
            )

    trained_channels = TRAINED_GRID * TRAINED_GRID
    ax_trend.axvline(
        trained_channels,
        linestyle="--",
        linewidth=1.5,
        color="0.45",
        zorder=1,
    )
    tick_channels = [n * n for n in ANNOTATED_GRIDS]
    ax_trend.set_xticks(tick_channels)
    ax_trend.set_xticklabels([f"{n * n}\n${n}\\times{n}$" for n in ANNOTATED_GRIDS])
    ax_trend.set_xticks([], minor=True)
    ax_trend.set_ylim(0.0, 0.2)
    ax_trend.set_xlim(80, 900)
    ax_trend.set_xlabel("number of channels")
    ax_trend.set_ylabel("relative $L_2$ error")

    tick_channels = [n * n for n in ANNOTATED_GRIDS]
    ax_trend.set_xticks(tick_channels)
    ax_trend.set_xticklabels([str(c) for c in tick_channels])
    ax_trend.set_xticks([], minor=True)
    ax_trend.grid(visible=True, axis="y", alpha=0.3)
    ax_trend.tick_params(labelsize=FONT["tick"])
    ax_trend.legend(fontsize=FONT["annotation"], frameon=False, loc="upper left")

    fig.suptitle(
        r"Trained on $15\times15$ only "
        r"($\theta = 45^\circ$, $\phi = 0^\circ$, $D = 7.63$ meV, $a = 3$ Å)",
        fontsize=FONT["suptitle"],
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "grid_generalisation_slide.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"--> Saved {out_path}")


if __name__ != "__main__":
    test_2()
