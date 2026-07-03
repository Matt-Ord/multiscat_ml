import contextlib
from pathlib import Path
from typing import Any, override

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
from multiscat.multiscat import get_scattering_matrix_from_state, get_scattering_state
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

PARAMS_MIN = np.array(
    [5.0, 0.5, 0.0, 0.02, 3, 3, 3, 8, 0.0, 0.0, 2, 3, 5, 5, 50],
    dtype=np.float64,
)
PARAMS_MAX = np.array(
    [10.0, 1.5, 4.0, 0.20, 6, 6, 6, 16, np.pi / 2, 2 * np.pi, 40, 10, 15, 15, 150],
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
        # Fixed-size inputs
        x_ds = f.create_dataset("X", shape=(num_samples, 15), dtype=np.float64)

        # Variable-size outputs go in a group
        y_group = f.create_group("Y")
        for i in range(num_samples):
            while True:
                try:
                    print(f"Generating sample {i + 1}/{num_samples}")

                    params = rng.uniform(size=15)
                    physical_params = denormalize_params(params)
                    Nx, Ny, Nz = map(int, np.round(physical_params[-3:]))
                    physical_params[-3:] = [Nx, Ny, Nz]
                    x = normalize_params(physical_params)
                    y = simulate_state(x)

                    x_ds[i] = x
                    y_group.create_dataset(str(i), data=y, dtype=np.float64)

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
        y_tensor = torch.from_numpy(self.file["Y"][str(idx)][()]).float()
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
    for i in range(20, 70):
        data_path = Path(f"data/15/state_data_{i}/.hdf5")
        generate_dataset_hdf5(data_path, num_samples=50)


def load_datasets() -> ConcatDataset[tuple[torch.Tensor, torch.Tensor]]:
    datasets = [
        HDF5ScatteringDataset(Path(f"data/15/state_data_{i}/.hdf5")) for i in range(70)
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


class ForwardStateModel(nn.Module):
    """Predicts state from 15 parameters using a deep ResNet at each grid."""

    def __init__(
        self,
        input_dim: int = 15,
        hidden_dim: int = 50,
        output_dim: int = 2,
    ) -> None:
        super().__init__()

        # 1. Expand the 6 parameters into a high-dimensional space
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
            nn.LeakyReLU(negative_slope=0.01),
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


def predict_chi_batch_from_params(
    forward_model: nn.Module,
    params_batch: torch.Tensor,
    chi_batch: list[torch.Tensor],
) -> list[torch.Tensor]:
    """
    Reconstruct a batch of chi grids from parameters using a pointwise forward model.

    Assumes:
        chi_batch shape  = (B, 2, Nx, Ny, Nz)
        params_batch shape = (B, 15)

    Returns
    -------
        chi_pred_batch shape = (B, 2, Nx, Ny, Nz)
    """
    device = params_batch.device
    chi_pred_batch: list[torch.Tensor] = []

    for ind, chi in enumerate(chi_batch):
        _, Nx, Ny, Nz = chi.shape

        xs = torch.linspace(0, 1, Nx, device=device)
        ys = torch.linspace(0, 1, Ny, device=device)
        zs = torch.linspace(0, 1, Nz, device=device)

        grid = torch.stack(
            torch.meshgrid(xs, ys, zs, indexing="ij"),
            dim=-1,
        )  # (Nx, Ny, Nz, 3)

        coords = grid.reshape(-1, 3)  # (Nx*Ny*Nz, 3)

        params = params_batch[ind].unsqueeze(0).repeat(coords.size(0), 1).clone()
        params[:, -3:] = coords

        pred = forward_model(params)  # (Nx*Ny*Nz, 2)

        pred = pred.view(Nx, Ny, Nz, 2).permute(3, 0, 1, 2).contiguous()
        chi_pred_batch.append(pred)

    return chi_pred_batch


def train() -> None:  # noqa: PLR0914, PLR0915
    dataset = load_datasets()
    train_dataset, val_dataset = random_split(dataset, [0.8, 0.2])

    def collate_variable(batch) -> tuple[torch.Tensor, list[torch.Tensor]]:
        xs, ys = zip(*batch, strict=False)

        xs = torch.stack(xs)  # X has fixed size
        ys = list(ys)  # Y has variable size
        return xs, ys

    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True, collate_fn=collate_variable
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, collate_fn=collate_variable
    )

    # 2. Initialize Models
    forward_model = ForwardStateModel().to(DEVICE)

    checkpoint = Path("data/15/best_chi_forward_model.pth")
    if checkpoint.exists():
        forward_model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
        print(f"Loaded pretrained model from {checkpoint}")
    else:
        print("No pretrained model found. Training from scratch.")

    # noqa:
    # backward_model = BackwardStateModel().to(DEVICE)

    # backward_criterion = nn.MSELoss()
    # maybe use TotalScatteringLoss(lambda_physics=0.01)
    forward_criterion = nn.MSELoss()
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
    patience = 15
    epochs_without_improvement = 0
    epochs = 100
    print(
        f"Training on {len(train_dataset)} samples,"
        f"Validating on {len(val_dataset)} samples...",
    )
    print(f"Using device: {DEVICE}")

    def mse_loss_for_tensor_lists(
        preds: list[torch.Tensor],
        targets: list[torch.Tensor],
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

        for params_batch, chi_batch in train_loader:
            params_batch = params_batch.to(DEVICE)  # noqa: PLW2901
            chi_batch = [chi.to(DEVICE) for chi in chi_batch]  # noqa: PLW2901
            chi_pred_batch = predict_chi_batch_from_params(
                forward_model, params_batch, chi_batch
            )
            loss_f = mse_loss_for_tensor_lists(
                chi_pred_batch, chi_batch, forward_criterion
            )
            loss_f.backward()
            forward_optimizer.step()

            train_loss_f += loss_f.item()

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

        with torch.no_grad():
            for params_batch, chi_batch in val_loader:
                params_batch = params_batch.to(DEVICE)  # noqa: PLW2901
                chi_batch = [chi.to(DEVICE) for chi in chi_batch]  # noqa: PLW2901

                # Forward Model Validation
                chi_pred_batch = predict_chi_batch_from_params(
                    forward_model, params_batch, chi_batch
                )
                val_loss_f += mse_loss_for_tensor_lists(
                    chi_pred_batch, chi_batch, forward_criterion
                ).item()

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
            f"Fwd Loss (Tr/Val): {average_train_loss_f:.1e} / {average_val_loss_f:.1e}"
            f"[LR: {lr_f:.1e}] | "
        )

        if average_val_loss_f < best_val_loss_f:
            best_val_loss_f = average_val_loss_f
            torch.save(forward_model.state_dict(), "data/15/best_chi_forward_model.pth")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print("Early stopping triggered!")
            break

    print("Training complete.")

    torch.save(forward_model.state_dict(), "data/15/chi_forward_model.pth")
    # torch.save(backward_model.state_dict(), "data/15/chi_backward_model.pth")


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
    forward_model = ForwardStateModel().to(DEVICE)
    forward_model.load_state_dict(
        torch.load("data/15/chi_forward_model.pth", map_location=DEVICE),
    )
    forward_model.eval()

    with torch.no_grad():
        channel_amp_dense_list = predict_chi_batch_from_params(
            forward_model,
            torch.tensor(test_params, dtype=torch.float32, device=DEVICE).unsqueeze(0),
            [torch.empty((2, *condition.metadata.shape), device=DEVICE)],
        )
        stat_data = channel_amp_dense_list[0][0] + 1j * channel_amp_dense_list[0][1]
        state = State(
            close_coupling_basis(condition.metadata).upcast(),
            stat_data.detach().cpu().numpy(),
        )
        pred_s_matrix = get_scattering_matrix_from_state(
            state, condition, n_channels=config.n_channels
        )

    fig, ax, _mesh = plot.array_against_axes_2d_k_nearest_neighbor(
        pred_s_matrix, measure="abs"
    )
    ax.set_title("Predicted scattering matrix")
    fig.savefig("data/15/scattering_matrix_from_predicted_state.png")

    actual = get_scattering_state(
        condition,
        config,
    )

    actual_s_matrix = get_scattering_matrix_from_state(
        actual, condition, n_channels=config.n_channels
    )

    fig, ax, _mesh = plot.array_against_axes_2d_k_nearest_neighbor(
        actual_s_matrix - pred_s_matrix, measure="abs"
    )
    fig.savefig("data/15/error_scattering_matrix_from_state.png")

    print(format_intensity_map(pred_s_matrix, threshold=1e-6))
    print("error intensity map:")
    print(format_intensity_map(actual_s_matrix - pred_s_matrix, threshold=1e-6))
    error = actual_s_matrix - pred_s_matrix
    print(np.sum(np.abs(error.raw_data)))

    fig, ax, _mesh = plot.array_against_axes_2d_k_nearest_neighbor(
        actual_s_matrix, measure="abs"
    )
    ax.set_title("The actual scattering matrix")
    fig.savefig("data/15/scattering_matrix_from_actual_state.png")


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
