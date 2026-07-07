import contextlib
import time
from pathlib import Path
from typing import Any, cast, override

import h5py  # type: ignore[import-untyped]
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
from slate_quantum import State, operator
from torch import nn, optim
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from tqdm import tqdm

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


def generate_dataset_hdf5(filepath: Path, num_samples: int = 50) -> None:
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

        x_tensor = torch.from_numpy(self.file["X"][idx]).float()
        y_tensor = torch.from_numpy(self.file["Y"][idx]).float()
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
    for i in range(10):
        data_path = Path(f"data/15/preconditioned_state_data_{i}.hdf5")
        generate_dataset_hdf5(data_path, num_samples=50)


def load_datasets() -> ConcatDataset[tuple[torch.Tensor, torch.Tensor]]:
    datasets = [
        HDF5ScatteringDataset(Path(f"data/15/preconditioned_state_data_{i}.hdf5"))
        for i in range(5)
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
        omega_0=30.0,
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
            nn.Linear(param_dim, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, cond_dim),
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
        omega_0: float = 30.0,
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
        first_omega_0: float = 30.0,
        hidden_omega_0: float = 1.0,
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

        x = coords
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


class BackwardStateModel(nn.Module):
    """Predicts parameters from one whole grid."""

    def __init__(self, nx: int, ny: int, nz: int) -> None:
        super().__init__()
        in_dim = 2 * nx * ny * nz

        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, 15),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 2, Nx, Ny, Nz)
        x = x.flatten(start_dim=1)  # (B, 2*Nx*Ny*Nz)
        return self.net(x)  # (B, 15)


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
        weighted_mse = torch.mean(base_error * weight_mask)

        # 3. Sparsity Penalty (L1)
        # This constantly applies a tiny downward pressure on all predicted values,
        # forcing the network to snap the background noise to exactly 0.0
        sparsity_loss = torch.mean(torch.abs(y_pred))

        return weighted_mse + (self.sparsity_weight * sparsity_loss)


def _make_coords(
    Nx: int,
    Ny: int,
    Nz: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    xs = torch.fft.fftfreq(Nx, device=device, dtype=dtype) * 2
    ys = torch.fft.fftfreq(Ny, device=device, dtype=dtype) * 2
    zs = torch.fft.fftfreq(Nz, device=device, dtype=dtype) * 2

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

    # Repeat parameters for each spatial point
    phys_params_rep = params_batch.unsqueeze(1).expand(B, N_pts, 12)  # (B, N_pts, 12)
    coords_rep = coords.unsqueeze(0).expand(B, N_pts, 3)  # (B, N_pts, 3)

    # Flatten into one big batch of points
    phys_params_flat = phys_params_rep.reshape(B * N_pts, 12)  # (B*N_pts, 12)
    coords_flat = coords_rep.reshape(B * N_pts, 3)  # (B*N_pts, 3)

    # Conditional model forward pass
    pred_flat = forward_model(phys_params_flat, coords_flat)  # (B*N_pts, 2)

    if pred_flat.ndim != 2 or pred_flat.shape[-1] != 2:
        msg = f"forward_model must return shape (B*N_pts, 2), got {pred_flat.shape}"
        raise ValueError(msg)

    # Reshape back to grid form
    pred_batch = pred_flat.view(B, N_pts, 2).transpose(1, 2).contiguous()

    return pred_batch.view(B, 2, Nx, Ny, Nz)


def train() -> None:  # noqa: PLR0914, PLR0915
    dataset = load_datasets()
    train_dataset, val_dataset = random_split(dataset, [0.8, 0.2])

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # 2. Initialize Models
    forward_model = ForwardCondSIRENStateModel().to(DEVICE)
    coord = _make_coords(Nx, Ny, Nz, device=DEVICE, dtype=torch.float32)

    checkpoint = Path("data/15/best_chi_SIREN_forward_model.pth")
    if checkpoint.exists():
        forward_model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
        print(f"Loaded pretrained model from {checkpoint}")
    else:
        print("No pretrained model found. Training from scratch.")

    # noqa:
    # backward_model = BackwardStateModel().to(DEVICE)

    # backward_criterion = nn.MSELoss()
    # maybe use TotalScatteringLoss(lambda_physics=0.01)
    forward_criterion = SparseScatteringLoss(peak_weight=10.0, sparsity_weight=1e-5)
    forward_optimizer = optim.AdamW(
        forward_model.parameters(),
        lr=1e-3,
        weight_decay=1e-5,
    )

    # backward_optimizer = optim.AdamW(
    #     backward_model.parameters(),
    #     lr=1e-3,
    #     weight_decay=1e-5,
    # )

    scheduler_f = optim.lr_scheduler.ReduceLROnPlateau(
        forward_optimizer,
        mode="min",
        factor=0.5,
        patience=5,
    )

    # scheduler_b = optim.lr_scheduler.ReduceLROnPlateau(
    #     backward_optimizer,
    #     mode="min",
    #     factor=0.5,
    #     patience=5,
    # )

    best_val_loss_f = float("inf")
    patience = 20
    epochs_without_improvement = 0
    epochs = 150
    print(
        f"Training on {len(train_dataset)} samples,"
        f"Validating on {len(val_dataset)} samples...",
    )
    print(f"Using device: {DEVICE}")

    def loss_for_tensor_lists(
        preds: torch.Tensor,
        targets: torch.Tensor,
        criterion: nn.Module | None = None,
    ) -> torch.Tensor:

        if len(preds) != len(targets):
            msg = f"Length mismatch: {len(preds)} preds vs {len(targets)} targets"
            raise ValueError(msg)

        if criterion is None:
            criterion = nn.MSELoss()

        loss = torch.zeros((), device=preds[0].device, dtype=preds[0].dtype)

        for pred, target in zip(preds, targets, strict=True):
            loss += criterion(pred, target)

        return loss / len(preds)

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

        # --- Backward Model Update (TANDEM ARCHITECTURE) ---
        # with freeze_parameters(forward_model):
        # backward_optimizer.zero_grad()
        # reconstructed_chi = forward_model(backward_model(chi_batch))
        # loss_b = backward_criterion(reconstructed_chi, chi_batch)
        # loss_b.backward()

        # backward_optimizer.step()
        # train_loss_b += loss_b.item()

        # --- VALIDATION PHASE ---
        forward_model.eval()
        # backward_model.eval()

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

                # Backward Model Validation (Tandem)
                # predicted = backward_model(chi_batch)
                # reconstructed_chi = forward_model(predicted)
                # val_loss_b += backward_criterion(
                #     reconstructed_chi,
                #     chi_batch,
                # ).item()

        # Averages
        average_train_loss_f = train_loss_f / len(train_loader)
        average_val_loss_f = val_loss_f / len(val_loader)
        # average_train_loss_b = train_loss_b / len(train_loader)
        # average_val_loss_b = val_loss_b / len(val_loader)

        # Step the schedulers
        scheduler_f.step(average_val_loss_f)
        # scheduler_b.step(average_val_loss_b)

        # Retrieve current learning rates for logging
        lr_f = forward_optimizer.param_groups[0]["lr"]
        # lr_b = backward_optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch + 1:03d}/{epochs} | "
            f"Fwd Loss (Tr/Val): {average_train_loss_f:.2e} / {average_val_loss_f:.2e}"
            f"[LR: {lr_f:.1e}] | "
        )

        if average_val_loss_f < best_val_loss_f:
            best_val_loss_f = average_val_loss_f
            torch.save(
                forward_model.state_dict(), "data/15/best_chi_SIREN_forward_model.pth"
            )
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print("Early stopping triggered!")
            break

    print("Training complete.")

    torch.save(forward_model.state_dict(), "data/15/chi_SIREN_forward_model.pth")
    # torch.save(backward_model.state_dict(), "data/15/chi_SIREN_backward_model.pth")


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
        torch.load("data/15/best_chi_SIREN_forward_model.pth", map_location=DEVICE),
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
        stat_data = channel_amp_dense[0] + 1j * channel_amp_dense[1]
        preconditioned_pred_state = State(
            close_coupling_basis(condition.metadata).upcast(),
            stat_data.detach().cpu().numpy(),
        )

    actual = get_scattering_state(
        condition,
        config,
    )

    preconditioned_state = get_preconditioned_state_from_state(
        actual, condition, n_channels=config.n_channels
    )

    # Return the real and imaginary parts of the preconditioned state
    preconditioned_actual_psi = preconditioned_state.with_basis(
        close_coupling_basis(condition.metadata)
    ).raw_data.reshape(condition.metadata.shape)[2, 3, :]

    precondtioned_pred_psi = preconditioned_pred_state.with_basis(
        close_coupling_basis(condition.metadata)
    ).raw_data.reshape(condition.metadata.shape)[2, 3, :]

    _, metadata_z = split_scattering_metadata(condition.metadata)
    height = metadata_z.domain.delta
    nz = condition.metadata.shape[2]
    z = np.linspace(0, height, nz)
    fig, ax1 = plot.get_figure()
    ax1.set_xlabel("z")
    ax1.set_ylabel(r"$\psi_{00}(z)$")
    ax1.plot(z, preconditioned_actual_psi.real, label="Actual real part")
    ax1.plot(z, precondtioned_pred_psi.real, label="Predicted real part")
    ax1.set_title("Actual and predicted scattering state")
    ax1.legend()
    fig.savefig("data/15/scattering_state.png")


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
