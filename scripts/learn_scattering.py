import contextlib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, override

import h5py  # type: ignore[import-untyped]
import numpy as np
import torch
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

# Constants
HELIUM_MASS = physical_constants["alpha particle mass"][0]
Z_HEIGHT = 8

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")  # pyright: ignore[reportConstantRedefinition]
else:
    DEVICE = torch.device("cpu")  # pyright: ignore[reportConstantRedefinition]

_BOUNDS = {
    "depth": (5.0, 10.0),
    "height": (0.5, 1.5),
    "offset": (2.0, 4.0),
    "beta": (0.05, 0.20),
    "a1x": (2.5, 5.0),
    "a1y": (2.5, 5.0),
    "a2x": (2.5, 5.0),
    "a2y": (2.5, 5.0),
    "theta": (0.0, np.pi / 2),
    "phi": (0.0, np.pi / 2),
    "energy": (1.0, 60.0),
}


@dataclass(frozen=True)
class ScatteringParams:
    # 11 Explicit Physical Parameters
    depth: float  # meV
    height: float  # Å
    offset: float  # Å
    beta: float  # unitless
    a1x: float  # Å
    a1y: float  # Å
    a2x: float  # Å
    a2y: float  # Å
    theta: float  # rad
    phi: float  # rad
    energy: float  # meV

    # Global hardware configuration limits [min, max] matching your ranges

    @classmethod
    def from_denormalized(cls, array: np.ndarray) -> ScatteringParams:
        """Build from a flat numpy array of unscaled physical parameters."""
        return cls(*array.astype(float))

    @classmethod
    def from_normalized(cls, array: np.ndarray) -> ScatteringParams:
        """Build from a flat numpy array of [0, 1] scaled parameters."""
        denormalized_values = []
        for val, f in zip(array, fields(cls), strict=False):
            p_min, p_max = _BOUNDS[f.name]
            denormalized_values.append(val * (p_max - p_min) + p_min)
        return cls(*denormalized_values)

    @classmethod
    def from_condition(
        cls,
        condition: MorseScatteringCondition,
    ) -> ScatteringParams:
        """Extract and construct parameters directly from a MorseScatteringCondition."""
        metadata_xy, _ = split_scattering_metadata(condition.metadata)

        a1 = (
            metadata_xy.extra.vectors[0][:2]
            * metadata_xy.children[0].domain.delta
            / angstrom_si
        )
        a2 = (
            metadata_xy.extra.vectors[1][:2]
            * metadata_xy.children[1].domain.delta
            / angstrom_si
        )

        return cls(
            depth=condition.morse_parameters.depth / (electron_volt * 10**-3),
            height=condition.morse_parameters.height / angstrom_si,
            offset=condition.morse_parameters.offset / angstrom_si,
            beta=condition.morse_parameters.beta,
            a1x=float(a1[0]),
            a1y=float(a1[1]),
            a2x=float(a2[0]),
            a2y=float(a2[1]),
            theta=condition.theta,
            phi=condition.phi,
            energy=condition.incident_energy / (electron_volt * 10**-3),
        )

    @property
    def denormalized(self) -> np.ndarray:
        """A flat 1D NumPy array of the raw physical units."""
        return np.array([getattr(self, f.name) for f in fields(self)], dtype=np.float64)

    @property
    def normalized(self) -> np.ndarray:
        """A flat 1D NumPy array scaled to a [0, 1] range."""
        norm_values = []
        for f in fields(self):
            p_min, p_max = _BOUNDS[f.name]
            val = getattr(self, f.name)
            norm_values.append((val - p_min) / (p_max - p_min))
        return np.array(norm_values, dtype=np.float64)

    @property
    def condition(self) -> MorseScatteringCondition:
        """Convert the current physical metrics into a MorseScatteringCondition object."""
        morse_params = operator.build.CorrugatedMorseParameters(
            depth=self.depth * electron_volt * 10**-3,
            height=self.height * angstrom_si,
            offset=self.offset * angstrom_si,
            beta=self.beta,
        )

        metadata = scattering_metadata_from_stacked_delta_x(
            (
                np.array([self.a1x * angstrom_si, self.a1y * angstrom_si, 0]),
                np.array([self.a2x * angstrom_si, self.a2y * angstrom_si, 0]),
                np.array([0, 0, Z_HEIGHT * angstrom_si]),
            ),
            (15, 15, 200),
        )

        return MorseScatteringCondition(
            mass=HELIUM_MASS,
            morse_parameters=morse_params,
            metadata=metadata,
            incident_k=momentum_from_angles(
                theta=self.theta,
                phi=self.phi,
                energy=self.energy * electron_volt * 10**-3,
                mass=HELIUM_MASS,
            ),
        )


def simulate_s_matrix(
    params: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    """Wrap your physics code into a single callable function."""
    condition = ScatteringParams.from_normalized(params).condition

    config = OptimizationConfig(precision=1e-5, max_iterations=1000, n_channels=80)
    s_matrix = get_scattering_matrix(condition, config, backend="scipy")

    metadata_x01, _ = split_scattering_metadata(condition.metadata)
    return s_matrix.with_basis(
        AsUpcast(basis.transformed_from_metadata(metadata_x01), metadata_x01),
    ).raw_data.real.reshape(15, 15)


def intensity_map_from_actual(
    actual: (Array[Any, np.dtype[np.complex128]]),
    *,
    threshold: float = 1e-8,
) -> list[tuple[int, int, float]]:
    """Convert scattering output into a sparse channel map of (kx, ky, intensity)."""
    data = actual.raw_data.reshape(actual.basis.metadata().shape)

    # Keep the FFT-style channel ordering used by multiscat plots.
    # For odd N: 0..N//2,-N//2..-1. For even N: 0..N/2-1,-N/2..-1.
    def _fft_channel_indices(size: int) -> list[int]:
        half = size // 2
        if size % 2 == 0:
            return [*range(half), *range(-half, 0)]
        return [*range(half + 1), *range(-half, 0)]

    nx, ny = data.shape
    kx_values = _fft_channel_indices(nx)
    ky_values = _fft_channel_indices(ny)

    rows: list[tuple[int, int, float]] = []
    for i, kx in enumerate(kx_values):
        for j, ky in enumerate(ky_values):
            intensity = float(np.abs(data[i, j]))
            if intensity > threshold:
                rows.append((int(kx), int(ky), intensity))

    return rows


def format_intensity_map(
    actual: (Array[Any, np.dtype[np.complex128]]),
    *,
    threshold: float = 1e-8,
) -> str:
    """Format scattering output as a text intensity map."""
    rows = intensity_map_from_actual(actual, threshold=threshold)
    lines = ["# kx ky intensity"]
    lines.extend(f"{kx:4d} {ky:4d}  {intensity:.8e}" for kx, ky, intensity in rows)
    return "\n".join(lines)


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


class SpecularInitLinear(nn.Linear):
    def reset_parameters(self) -> None:
        # Directly initialize to the desired map:
        # W = 0, b = [1, 0, 0, ..., 0]
        nn.init.zeros_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
            self.bias.data[0] = 1.0


class ForwardModel(nn.Module):
    def __init__(
        self, input_dim: int = 11, hidden_dim: int = 512, output_dim: int = 225
    ) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            ResBlock(hidden_dim),
            ResBlock(hidden_dim),
            ResBlock(hidden_dim),
            ResBlock(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
            nn.LeakyReLU(negative_slope=0.01),
        )

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).view(-1, 15, 15)


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
        input_data = f.create_dataset("X", shape=(num_samples, 11), dtype=np.float64)  # type: ignore[hd5]
        output_data = f.create_dataset(  # type: ignore[hd5]
            "Y",
            shape=(num_samples, 15, 15),
            dtype=np.float64,
        )

        for i in range(num_samples):
            print(f"Generating sample {i + 1}/{num_samples}")
            params = rng.uniform(size=11)

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
def freeze_parameters(model: nn.Module) -> Any:  # ruff: ignore[any-type]
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
        data_path = Path(f"data/15/scattering_dataset.{i}.hdf5")
        generate_dataset_hdf5(data_path, num_samples=1000)


def load_datasets() -> ConcatDataset[tuple[torch.Tensor, torch.Tensor]]:
    datasets = [
        HDF5ScatteringDataset(Path(f"data/15/scattering_dataset.{i}.hdf5"))
        for i in range(50)
    ]
    return ConcatDataset[tuple[torch.Tensor, torch.Tensor]](datasets)


def train() -> None:  # ruff: ignore[too-many-locals]
    dataset = load_datasets()
    train_dataset, val_dataset = random_split(dataset, [0.8, 0.2])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 2. Initialize Models
    forward_model = ForwardModel().to(DEVICE)
    forward_criterion = nn.MSELoss()
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

    best_val_loss_f = float("inf")
    patience = 15
    epochs_without_improvement = 0
    epochs = 100
    print(
        f"Training on {len(train_dataset)} samples,"
        f"Validating on {len(val_dataset)} samples...",
    )
    print(f"Using device: {DEVICE}")

    for epoch in range(epochs):
        forward_model.train()

        train_loss_f = 0.0

        for params_batch, s_mat_batch in train_loader:
            params_batch = params_batch.to(DEVICE)  # ruff: ignore[redefined-loop-name]
            s_mat_batch = s_mat_batch.to(DEVICE)  # ruff: ignore[redefined-loop-name]
            # --- Forward Model Update ---
            forward_optimizer.zero_grad()
            s_mat_pred = forward_model(params_batch)
            loss_f = forward_criterion(s_mat_pred, s_mat_batch)
            loss_f.backward()
            forward_optimizer.step()
            train_loss_f += loss_f.item()

        forward_model.eval()

        val_loss_f = 0.0

        with torch.no_grad():
            for params_batch, s_mat_batch in val_loader:
                params_batch = params_batch.to(DEVICE)  # ruff: ignore[redefined-loop-name]
                s_mat_batch = s_mat_batch.to(DEVICE)  # ruff: ignore[redefined-loop-name]

                # Forward Model Validation
                s_mat_pred = forward_model(params_batch)
                val_loss_f += forward_criterion(s_mat_pred, s_mat_batch).item()

        # Averages
        average_train_loss_f = train_loss_f / len(train_loader)
        average_val_loss_f = val_loss_f / len(val_loader)

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
            torch.save(forward_model.state_dict(), "data/15/best_forward_model.pth")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print("Early stopping triggered!")
            break

    print("Training complete.")

    torch.save(forward_model.state_dict(), "data/15/forward_model.pth")


def test() -> None:
    condition = MorseScatteringCondition(
        mass=HELIUM_MASS,
        morse_parameters=operator.build.CorrugatedMorseParameters(
            depth=7.63 * electron_volt * 10**-3,
            height=(1.0 / 1.1) * angstrom_si,
            offset=3.0 * angstrom_si,
            beta=0.05,
        ),
        metadata=scattering_metadata_from_stacked_delta_x(
            (
                np.array([8 * angstrom_si / np.sqrt(2), 0, 0]),
                np.array(
                    [8 * angstrom_si / np.sqrt(8), 8 * angstrom_si * np.sqrt(3 / 8), 0]
                ),
                np.array([0, 0, Z_HEIGHT * angstrom_si]),
            ),
            (15, 15, 200),
        ),
        incident_k=momentum_from_angles(
            theta=np.deg2rad(30),
            phi=np.deg2rad(0),
            energy=20 * electron_volt * 10**-3,
            mass=HELIUM_MASS,
        ),
    )

    test_params = torch.tensor(
        ScatteringParams.from_condition(condition).normalized,
        dtype=torch.float32,
    ).unsqueeze(0)  # Add batch dimension

    forward_model = ForwardModel().to(DEVICE)
    forward_model.load_state_dict(
        torch.load("data/15/forward_model.pth", map_location=DEVICE),
    )
    forward_model.eval()

    with torch.no_grad():
        channel_intensity_dense = forward_model(test_params)
        metadata_x01, _ = split_scattering_metadata(condition.metadata)
        predicted = Array(
            AsUpcast(basis.transformed_from_metadata(metadata_x01), metadata_x01),
            channel_intensity_dense.detach().cpu().numpy().astype(np.complex128),
        )
    fig, ax, _mesh = plot.array_against_axes_2d_k_nearest_neighbor(
        predicted, measure="abs"
    )
    ax.set_title("Predicted scattering matrix")
    fig.savefig("data/15/predicted_scattering_matrix.png")

    actual = get_scattering_matrix(
        condition,
        OptimizationConfig(precision=1e-5, max_iterations=1000, n_channels=49),
        backend="scipy",
    )

    fig, ax, _mesh = plot.array_against_axes_2d_k_nearest_neighbor(
        actual - predicted, measure="abs"
    )
    fig.savefig("data/15/error_scattering_matrix.png")

    print(format_intensity_map(predicted, threshold=1e-6))
    print("error intensity map:")
    print(format_intensity_map(actual - predicted, threshold=1e-6))
    error = actual - predicted
    print(np.sum(np.abs(error.raw_data)))

    fig, ax, _mesh = plot.array_against_axes_2d_k_nearest_neighbor(
        actual, measure="abs"
    )
    ax.set_title("The actual scattering matrix")
    fig.savefig("data/15/actual_scattering_matrix.png")


if __name__ == "__main__":
    generate()
    train()
    test()
