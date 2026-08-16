import pathlib
from pathlib import Path
from typing import TYPE_CHECKING, override

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
from scipy.interpolate import CubicSpline
from slate_core.plot import get_figure
from slate_quantum import operator
from torch import nn, optim
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    from pytorch_lightning.utilities.types import OptimizerLRSchedulerConfig


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")  # pyright: ignore[reportConstantRedefinition]
else:
    DEVICE = torch.device("cpu")  # pyright: ignore[reportConstantRedefinition]

PARAMS_MIN = np.array([0.1], dtype=np.float64)
PARAMS_MAX = np.array([2.0], dtype=np.float64)

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

    state = get_scattering_state(condition, config)
    preconditioned_state = get_preconditioned_state_from_state(
        state,
        condition,
        n_channels=config.n_channels,
    )

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


def get_flat_condition(condition: MorseScatteringCondition) -> MorseScatteringCondition:
    """Get a ScatteringCondition with zero corrugation (beta=0)."""
    return MorseScatteringCondition(
        mass=condition.mass,
        incident_k=condition.incident_k,
        morse_parameters=operator.build.CorrugatedMorseParameters(
            depth=condition.morse_parameters.depth,
            height=condition.morse_parameters.height,
            offset=condition.morse_parameters.offset,
            beta=0,
        ),
        metadata=condition.metadata,
        units=condition.units,
    )


def generate_stable_state_dataset_hdf5(
    filepath: Path, shape: tuple[int, int, int] = (18, 18, 200), n_samples: int = 500
) -> None:
    """Generate parameters and Preconditioned state, saving them directly to disk."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if filepath.exists():
        print(f"Dataset already exists at {filepath}. Skipping generation.")
        return

    print(f"Generating {n_samples} samples straight to disk. This may take a while...")
    rng = np.random.default_rng()

    with h5py.File(filepath, "w") as f:
        x_ds = f.create_dataset("X", shape=(n_samples, 1), dtype=np.float64)
        n_points = np.prod(shape)
        y_ds = f.create_dataset("Y", shape=(n_samples, n_points, 2), dtype=np.float64)

        for i in range(n_samples):
            print(f"Generating sample {i + 1}/{n_samples}")
            params = rng.uniform(size=1)
            condition = condition_from_params(params)
            config = OptimizationConfig(
                precision=1e-5, max_iterations=5000, n_channels=200
            )
            state_k_space = get_stable_state(condition, config)
            state_k_space = state_k_space.ravel()

            x_ds[i] = params
            y_ds[i] = np.column_stack([state_k_space.real, state_k_space.imag])


def sample_stable_state_dataset(
    params: np.ndarray[tuple[int], np.dtype[np.float64]],
    state_k_space: np.ndarray[tuple[int], np.dtype[np.complex128]],
    *,
    n_points: int = 100 * 100,
) -> tuple[
    np.ndarray[tuple[int], np.dtype[np.complex128]],
    np.ndarray[tuple[int, int], np.dtype[np.floating]],
    tuple[
        np.ndarray[tuple[int], np.dtype[np.complex128]],
        np.ndarray[tuple[int], np.dtype[np.complex128]],
        np.ndarray[tuple[int], np.dtype[np.complex128]],
    ],
]:
    """Sample state value and analytical spatial partial derivatives (dx, dy, dz)."""
    condition = condition_from_params(params)
    metadata_x01, metadata_z = split_scattering_metadata(condition.metadata)

    state_k_space = state_k_space.reshape(
        *metadata_x01.shape, metadata_z.fundamental_size
    )

    rng = np.random.default_rng()
    u = rng.uniform(0.0, 1.0, size=n_points)
    v = rng.uniform(0.0, 1.0, size=n_points)
    z = rng.uniform(metadata_z.values[0], metadata_z.values[-1], size=n_points)

    coords_out = np.stack(
        [
            np.cos(2 * np.pi * u),
            np.sin(2 * np.pi * u),
            np.cos(2 * np.pi * v),
            np.sin(2 * np.pi * v),
            z,
        ],
        axis=0,
    )

    # Construct wavevector grid in reciprocal space
    n_kx, n_ky = metadata_x01.shape
    m_freq = np.fft.fftfreq(n_kx) * n_kx
    n_freq = np.fft.fftfreq(n_ky) * n_ky

    # Calculate exponent phases and cubic spline
    phase = (2 * np.pi) * (
        m_freq[:, None, None] * u[None, None, :]
        + n_freq[None, :, None] * v[None, None, :]
    )
    exp_phase = np.exp(1j * phase)

    spline = CubicSpline(metadata_z.values, state_k_space, axis=2)
    spline_z = spline(z)
    spline_dz = spline.derivative(1)(z)

    # 1. Base wavefunction psi(x, y, z)
    psi = np.sum(spline_z * exp_phase, axis=(0, 1))

    # 2. Derivative d psi / du (x-direction)
    d_psi_du = np.sum(
        (1j * 2 * np.pi * m_freq[:, None, None]) * spline_z * exp_phase, axis=(0, 1)
    )

    # 3. Derivative d psi / dv (y-direction)
    d_psi_dv = np.sum(
        (1j * 2 * np.pi * n_freq[None, :, None]) * spline_z * exp_phase, axis=(0, 1)
    )

    # 4. Derivative d psi / dz (z-direction)
    d_psi_dz = np.sum(spline_dz * exp_phase, axis=(0, 1))

    return psi, coords_out, (d_psi_du, d_psi_dv, d_psi_dz)


def generate_sampled_dataset_hdf5(
    in_path: Path, out_path: Path, n_points: int = 100 * 100
) -> None:
    """Generate parameters, state values, and derivatives saving directly to disk."""
    if not in_path.exists():
        print(f"Dataset doesn't exist at {in_path}. Skipping generation.")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        print(f"Sampled dataset already exists at {out_path}. Skipping generation.")
        return

    with h5py.File(out_path, "w") as out_f, h5py.File(in_path, "r") as original_f:
        n_states = original_f["X"].shape[0]
        x_ds = out_f.create_dataset(
            "X", shape=(n_states, n_points, 6), dtype=np.float64
        )
        y_ds = out_f.create_dataset(
            "Y", shape=(n_states, n_points, 8), dtype=np.float32
        )

        for i in range(original_f["X"].shape[0]):
            print(f"Generating sample {i + 1}/{n_states}")

            params = original_f["X"][i]
            full_state = original_f["Y"][i, :, 0] + 1j * original_f["Y"][i, :, 1]

            state, coordinates, (d_psi_du, d_psi_dv, d_psi_dz) = (
                sample_stable_state_dataset(params, full_state, n_points=n_points)
            )

            energies = np.full((n_points, 1), params[0])
            x_ds[i] = np.hstack([coordinates.T, energies])
            y_ds[i] = np.column_stack(
                [
                    state.real,
                    state.imag,
                    d_psi_du.real,
                    d_psi_du.imag,
                    d_psi_dv.real,
                    d_psi_dv.imag,
                    d_psi_dz.real,
                    d_psi_dz.imag,
                ]
            ).astype(np.float32)


def generate_specular_dataset_hdf5(in_path: Path, out_path: Path) -> None:
    """Generate parameters and Preconditioned state, saving them directly to disk."""
    if not in_path.exists():
        print(f"Dataset doesn't exist at {in_path}. Skipping generation.")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        print(f"Specular dataset already exists at {out_path}. Skipping generation.")
        return

    with h5py.File(out_path, "w") as out_f, h5py.File(in_path, "r") as original_f:
        n_states = original_f["X"].shape[0]
        x_ds = out_f.create_dataset("X", shape=original_f["X"].shape, dtype=np.float64)
        y_ds = out_f.create_dataset("Y", shape=original_f["Y"].shape, dtype=np.float32)

        for i in range(original_f["X"].shape[0]):
            print(f"Generating sample {i + 1}/{n_states}")

            params = original_f["X"][i]
            full_state = original_f["Y"][i]

            condition = condition_from_params(params)
            config = OptimizationConfig(
                precision=1e-5, max_iterations=1000, n_channels=160
            )
            specular_condition = get_flat_condition(condition)
            specular_state = get_stable_state(specular_condition, config)
            specular_state = specular_state.ravel()
            specular_state = np.column_stack([specular_state.real, specular_state.imag])

            x_ds[i] = params
            y_ds[i] = full_state - specular_state


class HDF5ScatteringDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """PyTorch Dataset wrapper for reading (X, Y) pairs from a scattering HDF5 file."""

    def __init__(self, filepath: Path) -> None:
        self.filepath = filepath
        self._file = None

        with h5py.File(self.filepath, "r") as f:
            self._length = len(f["X"])

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:  # ty: ignore[invalid-method-override]
        if self._file is None:
            self._file = h5py.File(self.filepath, "r")

        x_arr = self._file["X"][idx]
        y_arr = self._file["Y"][idx]

        x_tensor = torch.from_numpy(x_arr).to(torch.float32)
        y_tensor = torch.from_numpy(y_arr).to(torch.float32)

        return x_tensor, y_tensor

    def __del__(self) -> None:
        if self._file is not None:
            self._file.close()


def generate_dataset() -> None:
    for i in range(50):
        data_path = Path(f"data/processed_state/stable_state_data_{i}.hdf5")
        generate_stable_state_dataset_hdf5(data_path, n_samples=500)


def generate_specular_dataset() -> None:
    for i in range(50):
        in_path = Path(f"data/processed_state/stable_state_data_{i}.hdf5")
        out_path = Path(f"data/processed_state/state_data_{i}.specular.hdf5")
        print(f"Generating specular dataset {i + 1}")
        generate_specular_dataset_hdf5(in_path, out_path)


def sample_dataset() -> None:
    for i in range(50):
        in_path = Path(f"data/processed_state/state_data_{i}.specular.hdf5")
        out_path = Path(f"data/processed_state/state_data_{i}.sampled.hdf5")
        generate_sampled_dataset_hdf5(in_path, out_path, n_points=100 * 100)


def load_datasets() -> ConcatDataset[tuple[torch.Tensor, torch.Tensor]]:
    files = sorted(Path("data/processed_state").glob("state_data_*.sampled.hdf5"))
    datasets = [HDF5ScatteringDataset(file) for file in files]
    return ConcatDataset[tuple[torch.Tensor, torch.Tensor]](datasets)


class ResBlock(nn.Module):
    """Pre-LayerNorm Residual Block optimized for smooth continuous field regression."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ComplexNN(nn.Module):
    def __init__(
        self,
        param_dim: int,
        hidden_dim: int = 128,
        num_blocks: int = 5,
    ) -> None:
        super().__init__()

        self.embedding = nn.Sequential(
            nn.Linear(param_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.res_blocks = nn.Sequential(
            *[ResBlock(hidden_dim) for _ in range(num_blocks)]
        )

        self.head = nn.Linear(hidden_dim, 2)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.res_blocks(x)
        return self.head(x)


class DerivativeLoss(nn.Module):
    """Derivative Loss calculated from spatial derivatives."""

    def __init__(self) -> None:
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(
        self, x: torch.Tensor, y_pred: torch.Tensor, y_true: torch.Tensor
    ) -> torch.Tensor:
        # 2. Compute gradients of predicted real and imag components w.r.t input x
        grad_real = torch.autograd.grad(
            outputs=y_pred[..., 0].sum(),
            inputs=x,
            create_graph=self.training,
            retain_graph=True,
            only_inputs=True,
        )[0]

        grad_imag = torch.autograd.grad(
            outputs=y_pred[..., 1].sum(),
            inputs=x,
            create_graph=self.training,
            retain_graph=True,
            only_inputs=True,
        )[0]

        # 3. Apply chain rule to reconstruct d/du, d/dv, d/dz
        # x0 = cos(2pi u), x1 = sin(2pi u) -> d/du = 2pi * (-x1 * d/dx0 + x0 * d/dx1)
        d_pred_du_r = -x[..., 1] * grad_real[..., 0] + x[..., 0] * grad_real[..., 1]
        d_pred_du_i = -x[..., 1] * grad_imag[..., 0] + x[..., 0] * grad_imag[..., 1]

        # x2 = cos(2pi v), x3 = sin(2pi v) -> d/dv = 2pi * (-x3 * d/dx2 + x2 * d/dx3)
        d_pred_dv_r = -x[..., 3] * grad_real[..., 2] + x[..., 2] * grad_real[..., 3]
        d_pred_dv_i = -x[..., 3] * grad_imag[..., 2] + x[..., 2] * grad_imag[..., 3]

        pred_derivatives = torch.stack(
            [
                2 * np.pi * d_pred_du_r,
                2 * np.pi * d_pred_du_i,
                2 * np.pi * d_pred_dv_r,
                2 * np.pi * d_pred_dv_i,
                # x4 = z -> d/dz = d/dx4
                grad_real[..., 4],
                grad_imag[..., 4],
            ],
            dim=-1,
        )

        return self.mse(pred_derivatives, y_true[..., 2:8])


class ScatteringLitModule(pl.LightningModule):
    def __init__(
        self,
        *,
        model: nn.Module,
        name: str,
        should_train: bool = True,
        base_path: Path = Path("data/processed_state"),
    ) -> None:
        super().__init__()
        self.model = model
        self.name = name
        self.should_train = should_train
        self.base_path = base_path
        self.save_hyperparameters(ignore=["model"])

        self.derivative_criterion = DerivativeLoss()
        self.mse_criterion = nn.MSELoss()

    @property
    def derivative_weight(self) -> float:
        """Linear ramp-up of relative derivative loss weight."""
        skip_epochs = 50
        ramp_epochs = 150
        max_weight = 0.005
        if self.current_epoch >= skip_epochs + ramp_epochs:
            return max_weight
        if self.current_epoch <= skip_epochs:
            return 0.0

        progress = (self.current_epoch - skip_epochs) / ramp_epochs
        return max_weight * (1 - np.cos(np.pi * progress)) / 2

    @property
    def checkpoint_path(self) -> Path:
        return self.base_path / self.name / "best_model.ckpt"

    @classmethod
    def load_or_initialize_model(
        cls,
        *,
        model: nn.Module,
        name: str,
        should_train: bool = True,
        base_path: Path = Path("data/processed_state"),
    ) -> ScatteringLitModule:
        if (base_path / name / "best_model.ckpt").exists():
            print(
                "loading best model from checkpoint",
                base_path / name / "best_model.ckpt",
            )
            torch.serialization.add_safe_globals(
                [pathlib.PosixPath, pathlib.WindowsPath]
            )
            return cls.load_from_checkpoint(
                checkpoint_path=base_path / name / "best_model.ckpt",
                model=model,
                should_train=should_train,
            )
        return cls(
            model=model, name=name, should_train=should_train, base_path=base_path
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], _batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        predictions = self(x)

        mse_loss = self.mse_criterion(predictions[..., 0:2], y[..., 0:2])

        if np.isclose(self.derivative_weight, 0.0):
            derivative_loss = torch.tensor(0.0, device=self.device)
        else:
            subsample_fraction = 0.1
            n_points = x.shape[1]
            subsample_size = max(1, int(n_points * subsample_fraction))

            idx = torch.rand(n_points, device=self.device).topk(subsample_size).indices

            x_sub = x[:, idx, :].detach().clone().requires_grad_()
            y_sub = y[:, idx, :]

            predictions_sub = self(x_sub)
            derivative_loss = self.derivative_criterion(x_sub, predictions_sub, y_sub)

        loss = mse_loss + self.derivative_weight * derivative_loss

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_mse_loss", mse_loss, on_step=False, on_epoch=True)
        self.log("train_derivative_loss", derivative_loss, on_step=False, on_epoch=True)

        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], _batch_idx: int
    ) -> None:
        x, y = batch
        predictions = self(x)
        mse_loss = self.mse_criterion(predictions[..., 0:2], y[..., 0:2])

        self.log("val_loss", mse_loss, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        optimizer = optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-4)
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
    data_fraction: float = 0.1,
) -> None:
    model_entry.base_path.mkdir(parents=True, exist_ok=True)

    sampled_dataset, _ = random_split(
        load_datasets(), [data_fraction, 1.0 - data_fraction]
    )
    train_dataset, val_dataset = random_split(sampled_dataset, [0.8, 0.2])

    train_loader = DataLoader(
        train_dataset, batch_size=8, shuffle=True, num_workers=0, pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=False
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=model_entry.base_path / model_entry.name,
        filename="best_model",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=max_epochs_without_improvement,
        mode="min",
    )
    csv_logger = CSVLogger(
        save_dir=model_entry.base_path,
        name=model_entry.name,
        version=0,
    )
    tb_logger = TensorBoardLogger(
        save_dir=model_entry.base_path,
        name=model_entry.name,
        version=1,
    )
    trainer = pl.Trainer(
        max_epochs=epochs,
        gradient_clip_val=1.0,
        callbacks=[checkpoint_callback, early_stop_callback],
        default_root_dir=model_entry.base_path,
        accelerator="auto",
        logger=[csv_logger, tb_logger],
        log_every_n_steps=10,
    )
    torch.serialization.add_safe_globals([pathlib.PosixPath, pathlib.WindowsPath])
    trainer.fit(
        model_entry,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=model_entry.checkpoint_path
        if model_entry.checkpoint_path.exists()
        else None,
    )


def get_predicted_stable_state(
    model: nn.Module,
    params: np.ndarray[tuple[int], np.dtype[np.float64]],
    grid_shape: tuple[int, int] = (80, 80),
) -> np.ndarray[tuple[int, int, int], np.dtype[np.complex128]]:
    """Predict the stable state from the given parameters using the trained model."""
    condition = condition_from_params(params)
    _, metadata_z = split_scattering_metadata(condition.metadata)

    u = np.linspace(0.0, 1.0, grid_shape[0], endpoint=False)
    v = np.linspace(0.0, 1.0, grid_shape[1], endpoint=False)

    cos_u_3d, sin_u_3d, cos_v_3d, sin_v_3d, z_3d = np.broadcast_arrays(
        np.cos(2 * np.pi * u[:, None, None]),
        np.sin(2 * np.pi * u[:, None, None]),
        np.cos(2 * np.pi * v[None, :, None]),
        np.sin(2 * np.pi * v[None, :, None]),
        metadata_z.values[None, None, :],
    )

    inputs_flat = np.stack(
        [
            cos_u_3d,
            sin_u_3d,
            cos_v_3d,
            sin_v_3d,
            z_3d,
            np.full_like(cos_u_3d, params[0]),
        ],
        axis=-1,
    ).reshape(-1, 6)

    device = next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        x_tensor = torch.from_numpy(inputs_flat).to(dtype=torch.float32, device=device)
        predictions = model(x_tensor).cpu().numpy()

    state = (predictions[:, 0] + 1j * predictions[:, 1]).reshape(*grid_shape, -1)
    return np.fft.fft2(state, axes=(0, 1), norm="forward")


def plot_channel_predictions(
    model_zoo: list[ScatteringLitModule],
    *,
    ax: plt.Axes | None = None,
    channel: tuple[int, int] = (0, 0),
) -> tuple[plt.Figure, plt.Axes]:
    rng = np.random.default_rng()
    test_params = rng.uniform(size=1)
    condition = condition_from_params(params=test_params)

    config = OptimizationConfig(precision=1e-5, max_iterations=1000, n_channels=160)
    actual = get_stable_state(condition, config=config)

    specular_condition = get_flat_condition(condition)
    specular_state = get_stable_state(specular_condition, config=config)
    actual -= specular_state

    fig, ax = get_figure(ax=ax)
    z_points = split_scattering_metadata(condition.metadata)[1].values
    actual_psi_00 = actual[*channel, :]
    (line1,) = ax.plot(z_points, np.real(actual_psi_00))
    (line2,) = ax.plot(z_points, np.imag(actual_psi_00))
    line2.set_color(line1.get_color())
    line2.set_linestyle("--")
    (line3,) = ax.plot(z_points, np.abs(actual_psi_00))
    line3.set_color(line1.get_color())
    line3.set_linestyle(":")

    for entry in model_zoo:
        predicted = get_predicted_stable_state(
            model=entry.model,
            params=test_params,
        )
        (line1,) = ax.plot(z_points, np.real(predicted[*channel, :]), label=entry.name)
        (line2,) = ax.plot(z_points, np.imag(predicted[*channel, :]))
        line2.set_color(line1.get_color())
        line2.set_linestyle("--")
        (line3,) = ax.plot(z_points, np.abs(predicted[*channel, :]))
        line3.set_color(line1.get_color())
        line3.set_linestyle(":")
    ax.set_xlabel("z")
    ax.set_ylabel(rf"$|\psi_{{{channel[0]}{channel[1]}}}(z)|$")
    ax.set_title(rf"Actual and predicted $\psi_{{{channel[0]}{channel[1]}}}(z)$")
    ax.legend()
    ax.grid(visible=True, alpha=0.25)

    return fig, ax


def plot_real_space_predictions(
    model_zoo: list[ScatteringLitModule],
    *,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    rng = np.random.default_rng()
    test_params = rng.uniform(size=1)
    condition = condition_from_params(params=test_params)

    config = OptimizationConfig(precision=1e-5, max_iterations=1000, n_channels=160)
    actual = get_stable_state(condition, config=config)

    specular_condition = get_flat_condition(condition)
    specular_state = get_stable_state(specular_condition, config=config)
    actual -= specular_state

    fig, ax = get_figure(ax=ax)
    z_points = split_scattering_metadata(condition.metadata)[1].values
    data_at_origin = np.fft.ifft2(actual, axes=(0, 1), norm="forward")[0, 0]
    (line1,) = ax.plot(z_points, np.real(data_at_origin))
    (line2,) = ax.plot(z_points, np.imag(data_at_origin))
    line2.set_color(line1.get_color())
    line2.set_linestyle("--")
    (line3,) = ax.plot(z_points, np.abs(data_at_origin))
    line3.set_color(line1.get_color())
    line3.set_linestyle(":")

    for entry in model_zoo:
        predicted = get_predicted_stable_state(
            model=entry.model,
            params=test_params,
        )
        data_at_origin = np.fft.ifft2(predicted, axes=(0, 1), norm="forward")[0, 0]
        (line1,) = ax.plot(z_points, np.real(data_at_origin), label=entry.name)
        (line2,) = ax.plot(z_points, np.imag(data_at_origin))
        line2.set_color(line1.get_color())
        line2.set_linestyle("--")
        (line3,) = ax.plot(z_points, np.abs(data_at_origin))
        line3.set_color(line1.get_color())
        line3.set_linestyle(":")

    ax.set_xlabel("z")
    ax.set_ylabel(r"$|\psi_{00}(z)|$")
    ax.set_title(r"Actual and predicted $\psi_{00}(z)$ for all models")
    ax.legend()
    ax.grid(visible=True, alpha=0.25)

    return fig, ax


def plot_energy_distribution(
    *,
    ax: plt.Axes | None = None,
    bins: int = 30,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot the distribution of energy factors in the training dataset."""
    energies = []
    for file in sorted(Path("data/processed_state").glob("state_data_*.sampled.hdf5")):
        with h5py.File(file, "r") as f:
            energies.append(f["X"][:, 0, -1])
    energies = np.array(energies).ravel()

    fig, ax = get_figure(ax=ax)
    ax.hist(energies, bins=bins, edgecolor="black", alpha=0.7)

    ax.set_xlabel("Energy Factor")
    ax.set_ylabel("Sample Count")
    ax.set_title("Training Set Energy Distribution")
    ax.grid(visible=True, alpha=0.25)

    return fig, ax


if __name__ == "__main__":
    if False:
        generate_dataset()
        generate_specular_dataset()
        sample_dataset()

    if False:
        fig, ax = plot_energy_distribution()
        fig.savefig("data/processed_state/energy_distribution.pdf")

    model_zoo: list[ScatteringLitModule] = [
        ScatteringLitModule(
            name="ComplexNN6",
            should_train=True,
            base_path=Path("data/processed_state"),
            model=ComplexNN(param_dim=6, hidden_dim=256, num_blocks=6),
        ),
    ]
    for m in model_zoo:
        if m.should_train:
            train_model(m, epochs=1000)

    fig, ax = plot_channel_predictions(model_zoo, channel=(0, 0))
    fig.savefig("data/processed_state/specular_predictions.pdf")

    fig, ax = plot_real_space_predictions(model_zoo)
    fig.savefig("data/processed_state/real_space_predictions.pdf")
