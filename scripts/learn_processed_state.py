from pathlib import Path
from typing import TYPE_CHECKING

import h5py  # type: ignore[import-untyped]
import numpy as np
import pytorch_lightning as pl
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
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from scipy.constants import (  # type: ignore[import-untyped]
    angstrom,
    electron_volt,
    physical_constants,
)
from slate_core.metadata.volume import fundamental_stacked_delta_x
from slate_core.plot import get_figure
from slate_quantum import operator
from torch import nn, optim
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    from pytorch_lightning.utilities.types import OptimizerLRSchedulerConfig


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
    files = sorted(Path("data/15").glob("stable_state_data_*.hdf5"))
    datasets = [HDF5ScatteringDataset(file) for file in files]
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


class PolarSIREN(nn.Module):
    def __init__(
        self,
        param_dim: int,
        hidden_dim: int,
        num_layers: int,
        omega_0: float = 1.0,
    ) -> None:
        super().__init__()

        layers = [
            SirenLayer(
                in_features=param_dim,
                out_features=hidden_dim,
                is_first=True,
                omega_0=omega_0,
            )
        ]

        layers.extend(
            SirenLayer(
                in_features=hidden_dim,
                out_features=hidden_dim,
                is_first=False,
                omega_0=omega_0,
            )
            for _ in range(num_layers - 1)
        )

        self.net = nn.ModuleList(layers)

        # Predict 2 outputs: [0] = magnitude, [1] = phase
        self.head = nn.Linear(hidden_dim, 2)
        self.init_head(omega_0)

    def init_head(self, omega_0: float) -> None:
        with torch.no_grad():
            bound = np.sqrt(6.0 / self.head.in_features) / omega_0
            self.head.weight.uniform_(-bound, bound)
            if self.head.bias is not None:
                self.head.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.net:
            x = layer(x)

        out = self.head(x)

        # 1. Separate magnitude and phase
        mag = torch.nn.functional.softplus(out[..., 0:1])
        phase = out[..., 1:2]

        # 2. Convert to real and imaginary
        real = mag * torch.cos(phase)
        imag = mag * torch.sin(phase)

        return torch.cat([real, imag], dim=-1)


# TODO: check we do have (x,y,z, *params)
# TODO: re scale x and y so periodicity is [0, 1]
class PeriodicComplexNN(nn.Module):
    def __init__(
        self,
        param_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_harmonics: int = 4,
    ) -> None:
        super().__init__()
        self.num_harmonics = num_harmonics

        # Assumes the first 2 dimensions of x are periodic (x, y)
        # Spatial dim = 2 coordinates * 2 (sin and cos) * num_harmonics
        spatial_dim = 2 * 2 * num_harmonics
        mlp_input_dim = spatial_dim + (param_dim - 2)

        # MLP backbone
        layers = []
        layers.extend((nn.Linear(mlp_input_dim, hidden_dim), nn.GELU()))

        for _ in range(num_layers - 1):
            layers.extend((nn.Linear(hidden_dim, hidden_dim), nn.GELU()))

        layers.append(nn.Linear(hidden_dim, 2))  # Predicts: [Real, Imag]
        self.mlp = nn.Sequential(*layers)

        # Frequencies for unit period (2 * pi * k * coord)
        harmonics = torch.arange(1, num_harmonics + 1, dtype=torch.float32)
        self.register_buffer("freqs", 2 * torch.pi * harmonics)

    def _fourier_features(self, coord: torch.Tensor) -> torch.Tensor:
        # coord shape: (..., 1)
        # freqs shape: (num_harmonics,)
        angles = coord * self.freqs  # (..., num_harmonics)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x shape: (..., param_dim)
        # x[..., 0:1] -> x coord, x[..., 1:2] -> y coord, x[..., 2:] -> remaining features (z, E, etc.)
        fx = self._fourier_features(x[..., 0:1])
        fy = self._fourier_features(x[..., 1:2])
        rest = x[..., 2:]

        # Concatenate spatial Fourier features with remaining dimensions
        features = torch.cat([fx, fy, rest], dim=-1)

        # Output shape: (..., 2) representing concatenated [real, imag]
        return self.mlp(features)


def get_predicted_stable_state(
    model: nn.Module,
    params: np.ndarray[tuple[int], np.dtype[np.float64]],
    grid_shape: tuple[int, int] = (80, 80),
) -> np.ndarray[tuple[int, int, int], np.dtype[np.complex128]]:
    """Predict the stable state from the given parameters using the trained model."""
    condition = condition_from_params(params)
    metadata_x01, metadata_z = split_scattering_metadata(condition.metadata)

    delta_x0, delta_x1 = fundamental_stacked_delta_x(metadata_x01)
    z_points = metadata_z.values

    N_kx, N_ky = grid_shape
    N_z = len(z_points)

    # 1. Generate an evenly spaced grid of fractional unit-cell coordinates [0, 1)
    u = np.linspace(0.0, 1.0, N_kx, endpoint=False)
    v = np.linspace(0.0, 1.0, N_ky, endpoint=False)
    u_2d, v_2d = np.meshgrid(u, v, indexing="ij")

    # 2. Map fractional coordinates to physical (x, y, z) positions
    x_grid = u_2d[..., None] * delta_x0[0] + v_2d[..., None] * delta_x1[0]
    y_grid = u_2d[..., None] * delta_x0[1] + v_2d[..., None] * delta_x1[1]
    z_grid = z_points[None, None, :]

    x_3d, y_3d, z_3d = np.broadcast_arrays(x_grid, y_grid, z_grid)

    # 3. Combine coordinates and energy_factor into model input array (N_total, 4)
    energy_grid = np.full_like(x_3d, params[0])
    inputs_flat = np.stack([x_3d, y_3d, z_3d, energy_grid], axis=-1).reshape(-1, 4)

    # 4. Predict real and imaginary parts using the neural network
    device = next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        x_tensor = torch.from_numpy(inputs_flat).to(dtype=torch.float32, device=device)
        predictions = model(x_tensor).cpu().numpy()

    state = (predictions[:, 0] + 1j * predictions[:, 1]).reshape(N_kx, N_ky, N_z)
    return np.fft.fft2(state, axes=(0, 1), norm="forward")


class ScatteringLitModule(pl.LightningModule):
    def __init__(
        self,
        *,
        model: nn.Module,
        name: str,
        train: bool = True,
        base_path: Path = Path("data/processed_state"),
    ) -> None:
        super().__init__()
        self.model = model
        self.criterion = nn.MSELoss()
        self.name = name
        self.should_train = train
        self.base_path = base_path

    @property
    def checkpoint_path(self) -> Path:
        return self.base_path / self.name / "best_model.ckpt"

    @classmethod
    def load_or_initialize_model(
        cls,
        *,
        model: nn.Module,
        name: str,
        train: bool = True,
        base_path: Path = Path("data/processed_state"),
    ) -> ScatteringLitModule:
        if (base_path / name / "best_model.ckpt").exists():
            return cls.load_from_checkpoint(
                checkpoint_path=base_path / name / "best_model.ckpt",
            )
        return cls(model=model, name=name, train=train, base_path=base_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], _batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        predictions = self(x)
        loss = self.criterion(predictions, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], _batch_idx: int
    ) -> None:
        x, y = batch
        predictions = self(x)
        loss = self.criterion(predictions, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        optimizer = optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }


def train_model(
    model_entry: ScatteringLitModule,
    epochs: int = 200,
    max_epochs_without_improvement: int = 200,
) -> None:
    model_entry.base_path.mkdir(parents=True, exist_ok=True)

    full_dataset = load_datasets()
    train_dataset, val_dataset, _ = random_split(full_dataset, [0.2, 0.05, 0.75])

    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=False
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=model_entry.base_path / model_entry.name,
        filename="best_model",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=max_epochs_without_improvement,
        mode="min",
    )
    csv_logger = CSVLogger(save_dir=model_entry.base_path, name=model_entry.name)
    # use with tensorboard --logdir data/
    tb_logger = TensorBoardLogger(save_dir=model_entry.base_path, name=model_entry.name)
    trainer = pl.Trainer(
        max_epochs=epochs,
        gradient_clip_val=1.0,
        callbacks=[checkpoint_callback, early_stop_callback],
        default_root_dir=model_entry.base_path,
        accelerator="auto",
        logger=[csv_logger, tb_logger],
        log_every_n_steps=10,
    )

    ckpt_path = model_entry.checkpoint_path
    trainer.fit(
        model_entry,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
    )


def plot_specular_predictions(
    model_zoo: list[ScatteringLitModule],
    *,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot actual and predicted specular prediction for every model."""
    test_params = np.array([1.0])
    condition = condition_from_params(params=test_params)

    config = OptimizationConfig(precision=1e-5, max_iterations=1000, n_channels=160)
    actual = get_stable_state(condition, config=config)

    fig, ax = get_figure(ax=ax)
    z_points = split_scattering_metadata(condition.metadata)[1].values
    actual_psi_00 = actual[0, 0, :]
    ax.plot(z_points, np.abs(actual_psi_00))

    for entry in model_zoo:
        predicted = get_predicted_stable_state(
            model=entry.model,
            params=test_params,
        )
        ax.plot(z_points, np.abs(predicted[0, 0, :]), label=entry.name)

    ax.set_xlabel("z")
    ax.set_ylabel(r"$|\psi_{00}(z)|$")
    ax.set_title(r"Actual and predicted $\psi_{00}(z)$ for all models")
    ax.legend()
    ax.grid(visible=True, alpha=0.25)

    return fig, ax


if __name__ == "__main__":
    # generate()

    model_zoo: list[ScatteringLitModule] = [
        ScatteringLitModule(
            name="PolarSIREN",
            train=False,
            base_path=Path("data/processed_state"),
            model=PolarSIREN(param_dim=4, omega_0=1.0, hidden_dim=256, num_layers=6),
        ),
        ScatteringLitModule(
            name="PeriodicComplexNN",
            train=False,
            base_path=Path("data/processed_state"),
            model=PeriodicComplexNN(
                param_dim=4, num_harmonics=15, hidden_dim=256, num_layers=6
            ),
        ),
    ]
    for m in model_zoo:
        if m.should_train:
            train_model(m, epochs=1000)

    fig, ax = plot_specular_predictions(model_zoo)
    fig.savefig("data/example_network/specular_predictions.pdf")
