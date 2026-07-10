import contextlib
import json
import time
from pathlib import Path
from typing import Any, cast, override

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
    get_scattering_matrix_from_preconditioned_state,
    get_scattering_state,
)
from multiscat.multiscat._scipy import (
    _build_scipy_operators,
)
from scipy.constants import angstrom as angstrom_si  # type: ignore[import-untyped]
from scipy.constants import (  # type: ignore[import-untyped]
    atomic_mass,
    electron_volt,
    physical_constants,
)
from slate_core import Array, metadata, plot
from slate_core.metadata import LobattoSpacedLengthMetadata
from slate_core.metadata._spaced import Domain
from slate_quantum import State, operator
from torch import nn, optim
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from tqdm import tqdm

a = LobattoSpacedLengthMetadata
# Constants
HELIUM_MASS = physical_constants["alpha particle mass"][0]
HELIUM_ENERGY = 20 * electron_volt * 10**-3
Z_HEIGHT = 8
Nx, Ny, Nz = 15, 15, 200

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")  # pyright: ignore[reportConstantRedefinition]
else:
    DEVICE = torch.device("cpu")  # pyright: ignore[reportConstantRedefinition]

PARAMS_MIN = np.array(
    [5.0, 0.5, 0.0, 0.02, 3, 3, 3, 8, 0.0, 0.0, 2, 3, 5, 5, 50],
    dtype=np.float64,
)
PARAMS_MAX = np.array(
    [10.0, 1.5, 4.0, 0.20, 6, 6, 6, 16, np.pi / 2, 2 * np.pi, 40, 10, Nx, Ny, Nz],
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
        depth,
        height,
        offset,
        beta,
        a1x,
        a2x,
        a2y,
        z_height,
        theta,
        phi,
        energy,
        mass,
        Nx,
        Ny,
        Nz,
    ) = denormalize_params(
        params,
    )

    Nx = round(Nx)
    Ny = round(Ny)
    Nz = round(Nz)

    morse_params = operator.build.CorrugatedMorseParameters(
        depth=depth * electron_volt * 10**-3,
        height=height * angstrom_si,
        offset=offset * angstrom_si,
        beta=beta,
    )

    Metadata = scattering_metadata_from_stacked_delta_x(
        (
            np.array([a1x * angstrom_si, 0, 0]),
            np.array([a2x * angstrom_si, a2y * angstrom_si, 0]),
            np.array([0, 0, z_height * angstrom_si]),
        ),
        (Nx, Ny, Nz),
    )

    return MorseScatteringCondition(
        mass=mass * atomic_mass,
        morse_parameters=morse_params,
        metadata=Metadata,
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

    a = metadata.volume.fundamental_stacked_delta_x(metadata_x01)
    a1x = a[0][0] / angstrom_si
    a2x, a2y = a[1][:2] / angstrom_si

    Nx, Ny, Nz = condition.metadata.shape

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
                depth,
                height,
                offset,
                beta,
                a1x,
                a2x,
                a2y,
                z_height,
                theta,
                phi,
                energy,
                mass,
                Nx,
                Ny,
                Nz,
            ],
        ),
    )


def simulate_state(
    params: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    """Wrap your physics code into a single callable function."""
    condition = condition_from_params(params)

    config = OptimizationConfig(precision=1e-5, max_iterations=1000, n_channels=250)
    state = get_scattering_state(condition, config)

    data = state.with_basis(
        close_coupling_basis(condition.metadata),
    ).raw_data.reshape(condition.metadata.shape)
    return np.array([data.real, data.imag])


def simulate_uppered_state(
    params: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    """Simulate the uppered state from the given parameters."""
    condition = condition_from_params(params)
    config = OptimizationConfig(precision=1e-5, max_iterations=1000, n_channels=250)
    converted_condition = _as_natural_units(condition)

    # Get the scattering state
    state = get_scattering_state(condition, config)

    # Get the preconditioned state
    preconditioned_state = get_preconditioned_state_from_state(
        state, condition, n_channels=config.n_channels
    )

    # Build the scipy operators
    _, _, upper = _build_scipy_operators(
        converted_condition, n_channels=config.n_channels
    )

    # Apply the upper operator to get the uppered state
    uppered_solution = upper.matvec(
        preconditioned_state.with_basis(
            close_coupling_basis(condition.metadata)
        ).raw_data
    ).reshape(condition.metadata.shape)

    # Convert the solution back to a State object
    uppered_state = State(
        close_coupling_basis(condition.metadata).upcast(),
        cast("np.ndarray[tuple[int], np.dtype[np.complex128]]", uppered_solution),
    )

    # Return the real and imaginary parts of the uppered state
    uppered_data = uppered_state.with_basis(
        close_coupling_basis(condition.metadata)
    ).raw_data.reshape(condition.metadata.shape)
    return np.array([uppered_data.real, uppered_data.imag])


def simulate_preconditioned_state(
    params: np.ndarray[tuple[int], np.dtype[np.float64]],
):
    """Simulate the preconditioned state from the given parameters."""
    condition = condition_from_params(params)
    config = OptimizationConfig(precision=1e-5, max_iterations=1000, n_channels=250)

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
    return torch.from_numpy(
        np.stack((preconditioned_data.real, preconditioned_data.imag), axis=0)
    ).float()


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
        x_ds = f.create_dataset("X", shape=(num_samples, 15), dtype=np.float64)

        # Variable-size outputs go in a group
        y_ds = f.create_dataset(
            "Y", shape=(num_samples, 2, Nx, Ny, Nz), dtype=np.float64
        )
        for i in range(num_samples):
            while True:
                try:
                    print(f"Generating sample {i + 1}/{num_samples}")

                    params = rng.uniform(size=15)
                    params[-3:] = 1
                    x_ds[i] = params
                    y_ds[i] = simulate_preconditioned_state(params)

                    break  # success

                except RuntimeError as e:
                    print(f"Sample {i + 1} failed ({e}), retrying...")


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
        Y_fft = torch.fft.fftshift(torch.fft.fft2(Y_complex, dim=(1, 2)), dim=(1, 2))

        # 4. Pack it back into a split Real/Imaginary view if your SIREN model expects 2 channels
        self.Y_data = torch.stack(
            [Y_fft.real, Y_fft.imag], dim=1
        )  # Shape: (B, 2, Nx, Ny, Nz)
        self.X_data = X_raw

        self.length = self.X_data.shape[0]
        print(f"--> Caching complete! Loaded {self.length} samples.")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Fast RAM slice—no disk reading overhead!
        return self.X_data[index], self.Y_data[index]


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
        data_path = Path(f"data/15/preconditioned_state_data_{i}.hdf5")
        generate_dataset_hdf5(data_path, num_samples=500)


def load_datasets() -> ConcatDataset[tuple[torch.Tensor, torch.Tensor]]:
    datasets = [
        HDF5ScatteringDataset(Path(f"data/15/preconditioned_state_data_{i}.hdf5"))
        for i in range(50)
    ]
    return ConcatDataset[tuple[torch.Tensor, torch.Tensor]](datasets)


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


class SirenLayer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        is_first=False,
        omega_0=15.0,
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

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class ConditionEncoder(nn.Module):
    """Encodes physical setup parameters into a latent conditioning vector."""

    def __init__(
        self,
        param_dim: int,
        cond_dim: int = 64,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(param_dim, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, cond_dim),
        )

    @override
    def forward(self, params: torch.Tensor) -> torch.Tensor:
        return self.net(params)


class FiLMSineLayer(nn.Module):
    """
    SIREN layer modulated by a conditioning vector.
    h = sin(omega_0 * ((Wx + b) * (1 + gamma(cond)) + beta(cond))).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        cond_dim: int,
        is_first: bool = False,
        omega_0: float = 15.0,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.gamma = nn.Linear(cond_dim, out_features)
        self.beta = nn.Linear(cond_dim, out_features)
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

            # Start close to an unmodulated SIREN
            nn.init.zeros_(self.gamma.weight)
            nn.init.zeros_(self.gamma.bias)
            nn.init.zeros_(self.beta.weight)
            nn.init.zeros_(self.beta.bias)

    @override
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.linear(x)
        gamma = self.gamma(cond)
        beta = self.beta(cond)
        h = h * (1.0 + gamma) + beta
        return torch.sin(self.omega_0 * h)


class ForwardCondSIRENStateModel(nn.Module):
    """
    Conditional SIREN for:
        (physical parameters, x, y, z) -> (Re(psi), Im(psi)).

    Recommended:
    - params: standardized physical inputs
    - coords: normalized to [-1, 1]
    """

    def __init__(
        self,
        param_dim: int = 12,  # e.g. depth, height, offset, beta, a1x, a2x, a2y, z_height, theta, phi, energy, mass
        coord_dim: int = 3,  # x, y, z
        cond_dim: int = 64,
        hidden_dim: int = 64,
        output_dim: int = 2,
        num_siren_layers: int = 4,
        first_omega_0: float = 15.0,
        hidden_omega_0: float = 15.0,
    ) -> None:
        super().__init__()

        self.condition_encoder = ConditionEncoder(
            param_dim=param_dim, cond_dim=cond_dim
        )

        layers = []
        layers.append(
            FiLMSineLayer(
                in_features=coord_dim,
                out_features=hidden_dim,
                cond_dim=cond_dim,
                is_first=True,
                omega_0=first_omega_0,
            )
        )
        layers.extend(
            FiLMSineLayer(
                in_features=hidden_dim,
                out_features=hidden_dim,
                cond_dim=cond_dim,
                is_first=False,
                omega_0=hidden_omega_0,
            )
            for _ in range(num_siren_layers - 1)
        )

        self.siren_layers = nn.ModuleList(layers)
        self.head = nn.Linear(hidden_dim, output_dim)

    @override
    def forward(self, params: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        params: (B, param_dim)
        coords: (B, 3)  normalized spatial coordinates, preferably in [-1, 1].
        """
        cond = self.condition_encoder(params)

        x = coords  # Shape: (N_pts, 3)
        for layer in self.siren_layers:
            x = layer(x, cond)

        return self.head(x)


class ForwardStateModel(nn.Module):
    """Predicts uppered state from 15 parameters using a deep ResNet at each grid."""

    def __init__(
        self, input_dim: int = 15, hidden_dim: int = 512, output_dim: int = 2
    ) -> None:
        super().__init__()

        # 1. Expand the 15 parameters into a high-dimensional space
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # 2. Process the features through multiple Residual Blocks
        # You can increase the number of blocks if the physics is highly complex
        self.res_blocks = nn.Sequential(
            ResBlock(hidden_dim),
            ResBlock(hidden_dim),
            ResBlock(hidden_dim),
            ResBlock(hidden_dim),
        )

        # 3. Collapse back down to the real and the imaginary parts of the 225 (15x15) chi-matrix
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
        )

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.res_blocks(x)
        out = self.head(x)
        return out.view(-1, 2)


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
        weight_mask = 1.0 + (self.peak_weight * torch.abs(y_true))

        # Apply the weight and take the mean
        return torch.mean(base_error * weight_mask)


class RelativePhysicalLoss(nn.Module):
    """
    Magnitude-invariant loss function tailored for continuous fields in position space.
    Normalizes errors by the local magnitude of the true signal.
    """

    def __init__(self, eps: float = 1e-4) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # Squared error at every single voxel
        squared_error = (y_pred - y_true) ** 2

        # Calculate the magnitude/intensity baseline for normalization at each voxel
        # y_true shape is (B, 2, Nx, Ny, Nz). We look at local intensity.
        true_magnitude = torch.abs(y_true)

        # Relative scaling: Normalize error by true magnitude + a small stabilizing floor (eps)
        relative_error = squared_error / (true_magnitude + self.eps)

        # Return the overall mean
        return torch.mean(relative_error)


class WavePhysicsLoss(nn.Module):
    def __init__(self, eps=1e-8) -> None:
        super().__init__()
        self.eps = eps
        self.cosine = nn.CosineSimilarity(dim=-1)  # Assumes flattened spatial grid

    def forward(self, pred, target):
        # 1. Relative L2 Loss (Captures balanced structural scaling)
        diff_norm = torch.norm(pred - target, p=2, dim=-1)
        target_norm = torch.norm(target, p=2, dim=-1)
        rel_l2 = torch.mean(diff_norm / (target_norm + self.eps))

        # 2. Phase Alignment Loss (Ensures peaks and troughs line up horizontally)
        # Cosine similarity outputs 1.0 for perfect alignment. We want to minimize (1 - similarity)
        phase_loss = torch.mean(1.0 - self.cosine(pred, target))

        # Combined Loss (Balanced 50/50 split)
        return rel_l2 + 1.0 * phase_loss


class AmplitudeAwarePhysicsLoss(nn.Module):
    def __init__(self, eps=1e-8) -> None:
        super().__init__()
        self.eps = eps
        self.cosine = nn.CosineSimilarity(dim=-1)

    def forward(self, pred, target):
        # 1. Compute the structural Relative L2 Error
        diff_norm = torch.norm(pred - target, p=2, dim=-1)
        target_norm = torch.norm(target, p=2, dim=-1)
        rel_l2 = torch.mean(diff_norm / (target_norm + self.eps))

        # 2. Compute the standard phase alignment
        phase_alignment = self.cosine(pred, target)
        phase_loss = torch.mean(1.0 - phase_alignment)

        # 3. The Amplitude Penalty (The Fix)
        # Compute the ratio of the predicted variance/energy vs true variance/energy
        pred_var = torch.var(pred, dim=-1)
        target_var = torch.var(target, dim=-1)
        amplitude_ratio_loss = torch.mean(
            torch.abs(pred_var - target_var) / (target_var + self.eps)
        )

        # Combine them: Total loss forces exact envelope height AND exact phase shifts
        return rel_l2 + 0.5 * phase_loss + 1.0 * amplitude_ratio_loss


def _make_coords(
    Nx: int,
    Ny: int,
    Nz: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    xs = torch.linspace(-1.0, 1.0, steps=Nx + 1, device=device, dtype=dtype)[:-1]
    ys = torch.linspace(-1.0, 1.0, steps=Ny + 1, device=device, dtype=dtype)[:-1]
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


def predict_chi_batch_from_params(
    forward_model: nn.Module,
    params_batch: torch.Tensor,
    coords: torch.Tensor,
    Nx: int,
    Ny: int,
    Nz: int,
    chunk_size: int = 20000,
) -> torch.Tensor:
    """
    Vectorized prediction of chi on a fixed grid for a batch of physical setups.

    Parameters
    ----------
    forward_model:
        Model with signature forward_model(params, coords) -> (B, 2) or (B*N_pts, 2)
        depending on how it is implemented.
    params_batch:
        Tensor of shape (B, 12), containing only the physical parameters.
    coords:
        Tensor of shape (N_pts, 3), normalized spatial coordinates.
    Nx, Ny, Nz:
        Grid dimensions used only for reshaping.

    Returns
    -------
    pred_batch:
        Tensor of shape (B, 2, Nx, Ny, Nz).
    """
    if params_batch.ndim != 2:
        msg = f"params_batch must have shape (B, 12), got {params_batch.shape}"
        raise ValueError(msg)
    if coords.ndim != 2 or coords.shape[-1] != 3:
        msg = f"coords must have shape (N_pts, 3), got {coords.shape}"
        raise ValueError(msg)
    if params_batch.shape[-1] != 12:
        msg = f"params_batch must have 12 physical parameters, got {params_batch.shape[-1]}"
        raise ValueError(msg)

    device = params_batch.device
    dtype = params_batch.dtype

    B = params_batch.shape[0]
    N_pts = coords.shape[0]

    coords = coords.to(device=device, dtype=dtype)

    # List to store the output chunks
    pred_chunks = []

    # Loop through the spatial points in safe, bite-sized pieces
    for i in range(0, N_pts, chunk_size):
        coords_chunk = coords[i : i + chunk_size]  # Shape: (chunk_N, 3)
        chunk_N = coords_chunk.shape[0]

        # Expand only this tiny chunk to match the batch size
        phys_params_rep = params_batch.unsqueeze(1).expand(
            B, chunk_N, 12
        )  # (B, chunk_N, 12)
        coords_rep = coords_chunk.unsqueeze(0).expand(B, chunk_N, 3)  # (B, chunk_N, 3)

        # Flatten just this chunk
        phys_params_flat = phys_params_rep.reshape(B * chunk_N, 12)
        coords_flat = coords_rep.reshape(B * chunk_N, 3)

        # Forward pass for the chunk (fits comfortably in VRAM!)
        pred_flat_chunk = forward_model(phys_params_flat, coords_flat)  # (B*chunk_N, 2)

        if pred_flat_chunk.ndim != 2 or pred_flat_chunk.shape[-1] != 2:
            msg = f"forward_model must return shape (B*N_pts, 2), got {pred_flat_chunk.shape}"
            raise ValueError(msg)

        # Reshape the chunk back to (B, chunk_N, 2)
        pred_chunk_reshaped = pred_flat_chunk.view(B, chunk_N, 2)
        pred_chunks.append(pred_chunk_reshaped)

    # Reconstruct the full spatial field along the points dimension
    full_pred = torch.cat(pred_chunks, dim=1)  # Shape: (B, N_pts, 2)

    # Rearrange dimensions and view as the requested 3D grid
    pred_batch = full_pred.transpose(1, 2).contiguous()
    return pred_batch.view(B, 2, Nx, Ny, Nz)


def plot_training_convergence(history: dict, save_path: Path) -> None:
    """Generates a publication-grade log-scale convergence plot."""
    # Use a clean aesthetic style
    plt.style.use(
        "seaborn-v0_8-whitegrid"
        if "seaborn-v0_8-whitegrid" in plt.style.available
        else "default"
    )

    _fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    epochs_range = range(1, len(history["train_loss"]) + 1)

    # Plot training and validation tracks
    ax.plot(
        epochs_range,
        history["train_loss"],
        label="Training Loss",
        color="#1f77b4",
        linewidth=2,
    )
    ax.plot(
        epochs_range,
        history["val_loss"],
        label="Validation Loss",
        color="#ff7f0e",
        linewidth=2,
        linestyle="--",
    )

    # Crucial scientific step: Logarithmic scale for wide dynamic ranges
    ax.set_yscale("log")

    # Labels and metadata
    ax.set_xlabel("Epochs", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("Loss (Log Scale)", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_title(
        "Model Convergence Profile Across Real Position Space",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )

    ax.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=11)
    ax.tick_params(axis="both", labelsize=10)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"--> Convergence plot saved to: {save_path}")


def train() -> None:  # noqa: PLR0914, PLR0915
    dataset = load_datasets()
    train_dataset, val_dataset = random_split(dataset, [0.8, 0.2])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 2. Initialize Models
    forward_model = ForwardCondSIRENStateModel().to(DEVICE)
    coord = _make_coords(Nx, Ny, Nz, device=DEVICE, dtype=torch.float32)

    checkpoint = Path("data/15/best_chi_SIREN_forward_model-FT-3.pth")
    if checkpoint.exists():
        forward_model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
        print(f"Loaded pretrained model from {checkpoint}")
    else:
        print("No pretrained model found. Training from scratch.")

    # noqa:
    forward_criterion = AmplitudeAwarePhysicsLoss()
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

    # 1. Initialize History Metrics Tracking Dictionary
    loss_history = {"train_loss": [], "val_loss": []}
    output_dir = Path("data/15")
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss_f = float("inf")
    patience = 20
    epochs_without_improvement = 0
    epochs = 200
    print(
        f"Training on {len(train_dataset)} samples,"
        f"Validating on {len(val_dataset)} samples...",
    )
    print(f"Using device: {DEVICE}")

    for epoch in range(epochs):
        forward_model.train()

        train_loss_f = 0.0
        # train_loss_b = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch")

        for batch_idx, (params_batch, preconditioned_state_batch) in enumerate(pbar):
            t0 = time.perf_counter()
            params_batch = params_batch.to(DEVICE)  # noqa: PLW2901
            preconditioned_state_batch = preconditioned_state_batch.to(DEVICE)  # noqa: PLW2901
            # --- Forward Model Update ---
            forward_optimizer.zero_grad()
            preconditioned_state_pred = predict_chi_batch_from_params(
                forward_model=forward_model,
                params_batch=params_batch[:, :12],
                coords=coord,
                Nx=Nx,
                Ny=Ny,
                Nz=Nz,
            )
            t1 = time.perf_counter()
            loss_f = forward_criterion(
                preconditioned_state_pred, preconditioned_state_batch
            )
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
            for batch_idx, (params_batch, preconditioned_state_batch) in enumerate(
                pbar_val
            ):
                params_batch = params_batch.to(DEVICE)  # noqa: PLW2901
                preconditioned_state_batch = preconditioned_state_batch.to(DEVICE)  # noqa: PLW2901

                # Forward Model Validation
                preconditioned_state_pred = predict_chi_batch_from_params(
                    forward_model=forward_model,
                    params_batch=params_batch[:, :12],
                    coords=coord,
                    Nx=Nx,
                    Ny=Ny,
                    Nz=Nz,
                )
                val_loss_f += forward_criterion(
                    preconditioned_state_pred, preconditioned_state_batch
                ).item()
                avg_val = val_loss_f / (batch_idx + 1)
                pbar_val.set_postfix(
                    loss=f"{forward_criterion(preconditioned_state_pred, preconditioned_state_batch).item():.3e}",
                    avg=f"{avg_val:.3e}",
                )

        # Averages
        average_train_loss_f = train_loss_f / len(train_loader)
        average_val_loss_f = val_loss_f / len(val_loader)

        # 2. Append history metrics at the end of every epoch
        loss_history["train_loss"].append(average_train_loss_f)
        loss_history["val_loss"].append(average_val_loss_f)

        # Step the schedulers
        scheduler_f.step(average_val_loss_f)

        # Retrieve current learning rates for logging
        lr_f = forward_optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch + 1:03d}/{epochs} | "
            f"Fwd Loss (Tr/Val): {average_train_loss_f:.2e} / {average_val_loss_f:.2e}"
            f"[LR: {lr_f:.1e}] | "
        )

        if average_val_loss_f < best_val_loss_f:
            best_val_loss_f = average_val_loss_f
            torch.save(
                forward_model.state_dict(),
                output_dir / "best_chi_SIREN_forward_model-FT-3.pth",
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

    torch.save(forward_model.state_dict(), "data/15/chi_SIREN_forward_model-FT-3.pth")


def test() -> None:
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
            energy=HELIUM_ENERGY,
            mass=HELIUM_MASS,
        ),
    )

    config = OptimizationConfig(precision=1e-5, max_iterations=1000, n_channels=250)

    test_params = params_from_condition(condition0)
    condition = condition_from_params(test_params)

    Nx, Ny, Nz = condition.metadata.shape
    coords = _make_coords(Nx, Ny, Nz, device=DEVICE, dtype=torch.float32)

    forward_model = ForwardCondSIRENStateModel().to(DEVICE)
    forward_model.load_state_dict(
        torch.load(
            "data/15/best_chi_SIREN_forward_model-FT-3.pth", map_location=DEVICE
        ),
    )
    forward_model.eval()

    with torch.no_grad():
        param_tensor = torch.tensor(
            test_params,
            dtype=torch.float32,
            device=DEVICE,
        ).unsqueeze(0)  # shape: (1, 15) or (1, 12), depending on your params

        channel_amp_dense_batch = predict_chi_batch_from_params(
            forward_model=forward_model,
            params_batch=param_tensor[:, :12],  # Only pass the physical parameters
            coords=coords,
            Nx=Nx,
            Ny=Ny,
            Nz=Nz,
        )

        channel_amp_dense = channel_amp_dense_batch[0]  # shape: (2, Nx, Ny, Nz)
        # 2. Combine channels into a complex tensor: Shape (Nx, Ny, Nz)
        spatial_data_complex = torch.complex(channel_amp_dense[0], channel_amp_dense[1])

        # 3. Return to Reciprocal Space (k-space)
        # Step A: Shift zero-frequencies back to the corners along X (dim 0) and Y (dim 1)
        unshifted_spatial = torch.fft.ifftshift(spatial_data_complex, dim=(0, 1))

        # Step B: Perform 2D IFFT over X and Y
        k_space_complex = torch.fft.ifft2(unshifted_spatial, dim=(0, 1))

        # Step C: SCALE BACK! Multiply by (Nx * Ny) to counteract PyTorch's default 1/N normalization
        k_space_complex *= 1

        # 4. Transfer to NumPy for your physical State class
        stat_data = k_space_complex.detach().cpu().numpy()
        preconditioned_pred_state = State(
            close_coupling_basis(condition.metadata).upcast(),
            stat_data,
        )

    actual = get_scattering_state(
        condition,
        config,
    )

    preconditioned_state = get_preconditioned_state_from_state(
        actual, condition, n_channels=config.n_channels
    )

    actual_k_space_np = preconditioned_state.with_basis(
        close_coupling_basis(condition.metadata)
    ).raw_data.reshape(condition.metadata.shape)
    actual_k_space = torch.from_numpy(actual_k_space_np).to(device=DEVICE)

    actual_spatial_complex = torch.fft.fftshift(
        torch.fft.fft2(actual_k_space, dim=(0, 1)), dim=(0, 1)
    )

    # Return the real and imaginary parts of the preconditioned state
    preconditioned_actual_psi = preconditioned_state.with_basis(
        close_coupling_basis(condition.metadata)
    ).raw_data.reshape(condition.metadata.shape)[0, 0, :]

    precondtioned_pred_psi = preconditioned_pred_state.with_basis(
        close_coupling_basis(condition.metadata)
    ).raw_data.reshape(condition.metadata.shape)[0, 0, :]

    _, metadata_z = split_scattering_metadata(condition.metadata)
    z = metadata_z.values
    fig, ax1 = plot.get_figure()
    ax1.set_xlabel("z")
    ax1.set_ylabel(r"$\psi_{00}(z)$")
    ax1.plot(z, preconditioned_actual_psi.real, label="Actual real part")
    ax1.plot(z, precondtioned_pred_psi.real, label="Predicted real part")
    ax1.set_title("Actual and predicted scattering state")
    ax1.legend()
    fig.savefig("data/15/scattering_state.png")

    actual_s_matrix = get_scattering_matrix_from_preconditioned_state(
        preconditioned_state, condition
    )
    predicted_s_matrix = get_scattering_matrix_from_preconditioned_state(
        preconditioned_pred_state, condition
    )

    fig, ax, _mech = plot.array_against_axes_2d_k_nearest_neighbor(
        actual_s_matrix, measure="abs"
    )
    ax.set_title("The actual scattering matrix")
    fig.savefig("data/15/scattering_matrix_from_actual_state.png")

    fig, ax, _mech = plot.array_against_axes_2d_k_nearest_neighbor(
        predicted_s_matrix, measure="abs"
    )
    ax.set_title("The predicted scattering matrix")
    fig.savefig("data/15/scattering_matrix_from_predicted_state.png")

    # =====================================================================
    # 1. Extract 1D Slices at the Center (x=0, y=0) as a function of z
    # =====================================================================
    center_x = Nx // 2
    center_y = Ny // 2

    # Get the 1D complex arrays along the Z-axis
    pred_psi_z_complex = spatial_data_complex[center_x, center_y, :]
    actual_psi_z_complex = actual_spatial_complex[center_x, center_y, :]

    # Calculate absolute values (magnitudes) and convert to NumPy
    pred_psi_z_abs = torch.abs(pred_psi_z_complex).cpu().numpy()
    actual_psi_z_abs = torch.abs(actual_psi_z_complex).cpu().numpy()

    # =====================================================================
    # 2. Plotting the Real Space Amplitude Comparison
    # =====================================================================
    fig, ax1 = plot.get_figure()
    ax1.set_xlabel("z", fontsize=11, fontweight="bold")
    ax1.set_ylabel(
        r"$|\psi(x=0, y=0, z)|$ (Real Space)", fontsize=11, fontweight="bold"
    )

    # Plot the absolute magnitudes
    ax1.plot(
        z, actual_psi_z_abs, label="Actual Amplitude", color="#1f77b4", linewidth=2
    )
    ax1.plot(
        z,
        pred_psi_z_abs,
        label="Predicted Amplitude",
        color="#ff7f0e",
        linestyle="--",
        linewidth=2,
    )

    ax1.set_title(
        "Real Space Scattering Amplitude Profile down the Center Axis",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax1.legend(frameon=True, facecolor="white")

    # Save to your directory
    fig.savefig(
        "data/15/scattering_amplitude_real_space.png", bbox_inches="tight", dpi=300
    )
    print("--> Real space 1D amplitude slice plot saved successfully.")


if __name__ == "__main__":
    RUN_GENERATE = True
    RUN_TRAIN = True
    RUN_TEST = True

    if RUN_GENERATE:
        generate()
    if RUN_TRAIN:
        train()
    if RUN_TEST:
        test()
