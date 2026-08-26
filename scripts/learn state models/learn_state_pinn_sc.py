from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, override

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
    _as_natural_units,  # ruff: ignore[import-private-name]
    get_preconditioned_state_from_state,
    get_scattering_matrix_from_preconditioned_state,
    get_scattering_state,
)
from multiscat.multiscat._scipy import (  # ruff: ignore[import-private-name]
    _build_scipy_operators,
)
from multiscat.multiscat._util import (  # ruff: ignore[import-private-name]
    get_target_state,
)
from scipy.constants import angstrom as angstrom_si  # type: ignore[import-untyped]
from scipy.constants import (  # type: ignore[import-untyped]
    electron_volt,
    physical_constants,
)
from slate_core import Array, metadata, plot
from slate_core.metadata import LobattoSpacedLengthMetadata
from slate_core.metadata._spaced import Domain  # ruff: ignore[import-private-name]
from slate_quantum import State, operator
from torch import nn, optim
from tqdm import tqdm

# Constants
HELIUM_MASS = physical_constants["alpha particle mass"][0]
HELIUM_ENERGY = 20 * electron_volt * 10**-3
Z_HEIGHT = 8
Nx, Ny, Nz = 9, 9, 100

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")  # pyright: ignore[reportConstantRedefinition]
else:
    DEVICE = torch.device("cpu")  # pyright: ignore[reportConstantRedefinition]

PARAMS_MIN = np.array(
    [5.0, 0.5, 0.0, 0.02, 2.0, 0.0],
    dtype=np.float64,
)
PARAMS_MAX = np.array(
    [10.0, 1.5, 4.0, 0.20, 4.0, np.pi * 2 / 5],
    dtype=np.float64,
)


def denormalize_params(params_norm: torch.Tensor | np.ndarray) -> torch.Tensor:
    """Scales parameters back to their original physical units.

    Supports input as a NumPy array or PyTorch Tensor (CPU or CUDA).
    """
    if isinstance(params_norm, torch.Tensor):
        # Convert static bounds to torch Tensors matching the input device and dtype
        p_max = torch.as_tensor(
            PARAMS_MAX, device=params_norm.device, dtype=params_norm.dtype
        )
        p_min = torch.as_tensor(
            PARAMS_MIN, device=params_norm.device, dtype=params_norm.dtype
        )
        return params_norm * (p_max - p_min) + p_min

    # Fallback to standard NumPy path
    return params_norm * (PARAMS_MAX - PARAMS_MIN) + PARAMS_MIN


def condition_from_params(
    params: torch.Tensor | np.ndarray,
) -> MorseScatteringCondition:
    denorm = denormalize_params(params)
    if isinstance(denorm, torch.Tensor):
        denorm_np = denorm.detach().cpu().numpy()
    else:
        denorm_np = np.asarray(denorm)
    (
        depth,
        height,
        offset,
        beta,
        a,
        theta,
    ) = denorm_np
    morse_params = operator.build.CorrugatedMorseParameters(
        depth=depth * electron_volt * 10**-3,
        height=height * angstrom_si,
        offset=offset * angstrom_si,
        beta=beta,
    )

    Metadata = scattering_metadata_from_stacked_delta_x(  # ruff: ignore[non-lowercase-variable-in-function]
        (
            np.array([a * angstrom_si, 0, 0]),
            np.array([0, a * angstrom_si, 0]),
            np.array([0, 0, Z_HEIGHT * angstrom_si]),
        ),
        (Nx, Ny, Nz),
    )

    return MorseScatteringCondition(
        mass=HELIUM_MASS,
        morse_parameters=morse_params,
        metadata=Metadata,
        incident_k=momentum_from_angles(
            theta=theta,
            phi=0.0,
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

    a_vec = metadata.volume.fundamental_stacked_delta_x(metadata_x01)
    a = a_vec[0][0] / angstrom_si

    morse_parameters = condition.morse_parameters

    depth = morse_parameters.depth / (electron_volt * 10**-3)
    height = morse_parameters.height / angstrom_si
    offset = morse_parameters.offset / angstrom_si
    beta = morse_parameters.beta
    theta = condition.theta

    return normalize_params(
        np.array(
            [depth, height, offset, beta, a, theta],
        ),
    )


def pack_complex(x: np.ndarray) -> np.ndarray:
    """
    Convert a complex vector/array into a 2-channel real array.

        out[0] = Re(x)
        out[1] = Im(x).
    """
    x = np.asarray(x)
    return np.stack([x.real, x.imag], axis=0)


def unpack_complex(x_2ch: np.ndarray) -> np.ndarray:
    """Convert a 2-channel real array [Re(x), Im(x)] back to complex."""
    x_2ch = np.asarray(x_2ch, dtype=np.float64)
    return x_2ch[0] + 1j * x_2ch[1]


def simulate_target_state(
    params: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    """
    Return the target state in packed real/imag form.

        shape = (2, Nx, Ny, Nz).
    """
    condition = _as_natural_units(condition_from_params(params))
    inverse_lower, *_ = _build_scipy_operators(condition, n_channels=250)
    state = get_target_state(condition.metadata, condition.incident_k)
    raw = state.with_basis(close_coupling_basis(condition.metadata)).raw_data
    processed = inverse_lower.matvec(raw)
    data = processed.reshape(condition.metadata.shape)
    return pack_complex(data)


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


class PureMLP(nn.Module):
    def __init__(  # ruff: ignore[too-many-arguments, too-many-positional-arguments]
        self,
        param_dim: int = 6,
        coord_dim: int = 3,
        hidden_dim: int = 1024,
        num_blocks: int = 8,
        output_dim: int = 2,
        dropout_rate: float = 0.05,
    ) -> None:
        """Map the coordinates (n, m, z) to the wavefunction at the point for a pointwise flexible mapping.

        Plain MLP is used for simplicity, but the model is currently laboured under severe spectral bias
        """
        super().__init__()

        self.param_dim = param_dim
        self.coord_dim = coord_dim
        self.output_dim = output_dim

        in_dim = param_dim + coord_dim
        self.embedding = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.res_blocks = nn.Sequential(
            *[ResBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)]
        )

        self.head = nn.Linear(hidden_dim, output_dim)

        self.init_weights()

    def init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    @override
    def forward(
        self,
        params: torch.Tensor,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([params, coords], dim=-1)
        x = self.embedding(x)
        x = self.res_blocks(x)

        return self.head(x)


def _make_coords(
    nx: int,
    ny: int,
    nz: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    xs = torch.fft.fftfreq(nx, d=1.0 / nx).to(device=device, dtype=dtype)
    ys = torch.fft.fftfreq(ny, d=1.0 / ny).to(device=device, dtype=dtype)
    z_domain = Domain(start=0.0, delta=Z_HEIGHT)
    z_metadata = LobattoSpacedLengthMetadata(fundamental_size=nz, domain=z_domain)
    zs_np = z_metadata.values
    zs = torch.from_numpy(zs_np).to(device=device, dtype=dtype)

    # Create the 3D grid
    grid = torch.stack(
        torch.meshgrid(xs, ys, zs, indexing="ij"),
        dim=-1,
    )  # (Nx, Ny, Nz, 3)

    return grid.reshape(-1, 3)  # (N, 3)


def predict_chi_batch_from_params(  # ruff: ignore[too-many-arguments]  # ruff: ignore[too-many-positional-arguments]
    forward_model: nn.Module,
    params_batch: torch.Tensor,
    coords: torch.Tensor,
    nx: int,
    ny: int,
    nz: int,
) -> torch.Tensor:
    b = params_batch.shape[0]
    n_pts = coords.shape[0]

    coords = coords.to(device=params_batch.device, dtype=params_batch.dtype)

    # Broadcast: every parameter vector against every spatial point
    params_grid = params_batch.unsqueeze(1).expand(b, n_pts, -1)
    coords_grid = coords.unsqueeze(0).expand(b, n_pts, -1)

    pred = forward_model(
        params=params_grid,
        coords=coords_grid,
    )  # (B, N_pts, output_dim)

    return pred.view(b, nx, ny, nz, -1).permute(0, 4, 1, 2, 3)


class ApplyLUFn(torch.autograd.Function):
    """Define the lhs action: (1 + L^-1 U) psi in the linear equation the GMRES solves.

    This class also defines the adjoint action for the back propagation in the gradient descent calculation for the training.
    """

    @staticmethod
    def forward(  # ruff: ignore[too-many-arguments]  # ruff: ignore[too-many-positional-arguments]
        ctx: Any,  # ruff: ignore[any-type]
        pred_batch: torch.Tensor,
        params_batch: torch.Tensor,
        nx: int,
        ny: int,
        nz: int,
    ) -> torch.Tensor:
        device = pred_batch.device
        dtype = pred_batch.dtype

        lhs_list = []
        ops_list = []

        for b in range(pred_batch.shape[0]):
            params_np = params_batch[b].detach().cpu().numpy()
            condition = _as_natural_units(condition_from_params(params_np))
            inverse_lower, _lower, upper = _build_scipy_operators(
                condition, n_channels=250
            )

            chi_real = pred_batch[b, 0].reshape(-1).detach().cpu().numpy()
            chi_imag = pred_batch[b, 1].reshape(-1).detach().cpu().numpy()
            chi = chi_real + 1j * chi_imag

            lhs = chi + inverse_lower.matvec(upper.matvec(chi))

            lhs_realimag = torch.stack(
                [
                    torch.from_numpy(np.real(lhs)).to(device=device, dtype=dtype),
                    torch.from_numpy(np.imag(lhs)).to(device=device, dtype=dtype),
                ],
                dim=0,
            ).reshape(2, nx, ny, nz)

            lhs_list.append(lhs_realimag)
            ops_list.append((inverse_lower, upper))

        ctx.ops_list = ops_list
        ctx.Nx = nx
        ctx.Ny = ny
        ctx.Nz = nz
        return torch.stack(lhs_list, dim=0)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx: Any, grad_output: torch.Tensor) -> Any:  # type: ignore[override] # ruff: ignore[any-type]
        grad_pred = torch.zeros_like(grad_output)

        for b, (inverse_lower, upper) in enumerate(ctx.ops_list):
            grad_real = grad_output[b, 0].reshape(-1).detach().cpu().numpy()
            grad_imag = grad_output[b, 1].reshape(-1).detach().cpu().numpy()
            g = grad_real + 1j * grad_imag

            g_in = g + upper.H.matvec(inverse_lower.H.matvec(g))

            grad_pred[b, 0] = (
                torch.from_numpy(np.real(g_in))
                .to(grad_output.device, grad_output.dtype)
                .reshape(ctx.Nx, ctx.Ny, ctx.Nz)
            )
            grad_pred[b, 1] = (
                torch.from_numpy(np.imag(g_in))
                .to(grad_output.device, grad_output.dtype)
                .reshape(ctx.Nx, ctx.Ny, ctx.Nz)
            )

        return grad_pred, None, None, None, None


def save_loss_history(loss_history: dict, path: str | Path) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(loss_history, f, indent=4)


def plot_training_convergence(history: dict, save_path: Path) -> None:
    """Generate a log-scale convergence plot."""
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
        "Model Convergence Profile",
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


def _sample_params_batch(rng: np.random.Generator, batch_size: int) -> np.ndarray:
    return rng.uniform(size=(batch_size, 6))


def _make_target_batch(params_batch_np: np.ndarray) -> np.ndarray:
    targets = [simulate_target_state(params) for params in params_batch_np]
    return np.stack(targets, axis=0)


def _make_fixed_val_set(
    rng: np.random.Generator,
    num_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    params_np = _sample_params_batch(rng, num_samples)
    target_np = _make_target_batch(params_np)

    params = torch.from_numpy(params_np).float()
    target = torch.from_numpy(target_np).float()
    return params, target


def train_rand(  # ruff: ignore[too-many-statements] # ruff: ignore[too-many-arguments]  # ruff: ignore[too-many-positional-arguments]   # ruff: ignore[too-many-locals]
    output_dir: str | Path = "data/state_model",
    epochs: int = 600000,
    patience: int = 600000,
    train_batch_size: int = 4,
    val_batch_size: int = 4,
    steps_per_epoch: int = 1,
    num_val_samples: int = 32,
    resume: bool = True,  # ruff: ignore[boolean-type-hint-positional-argument]   #ruff: ignore[boolean-default-value-positional-argument]
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = output_dir / "best_chi_model.pth"
    final_model_path = output_dir / "chi_model.pth"
    loss_history_path = output_dir / "loss_history.json"
    plot_path = output_dir / "convergence_curve.png"

    OptimizationConfig(precision=1e-5, max_iterations=1000, n_channels=250)

    forward_model = PureMLP().to(DEVICE)
    if resume and best_model_path.exists():
        forward_model.load_state_dict(
            torch.load(best_model_path, map_location=DEVICE),
        )
        print(f"--> Resumed weights from: {best_model_path}")

    coords = _make_coords(Nx, Ny, Nz, device=DEVICE, dtype=torch.float32)

    forward_criterion = nn.MSELoss()
    forward_optimizer = optim.AdamW(
        forward_model.parameters(),
        lr=1e-6,
        weight_decay=1e-5,
    )

    scheduler_f = optim.lr_scheduler.ReduceLROnPlateau(
        forward_optimizer,
        mode="min",
        factor=0.5,
        patience=5,
    )

    # Fixed validation set for comparable metrics across epochs
    rng_val = np.random.default_rng(12345)
    val_params_all, val_target_all = _make_fixed_val_set(rng_val, num_val_samples)

    # Training RNG
    rng_train = np.random.default_rng()

    loss_history = {"train_loss": [], "val_loss": []}
    best_val_loss_f = float("inf")
    epochs_without_improvement = 0
    start_epoch = 0

    # Current-epoch running state
    train_loss_sum = 0.0
    train_batches_done = 0
    val_loss_sum = 0.0
    val_batches_done = 0

    print(f"Using device: {DEVICE}")

    for epoch in range(start_epoch, epochs):
        train_start = 0
        val_start = 0

        # -------------------------
        # TRAINING PHASE
        # -------------------------
        forward_model.train()

        pbar = tqdm(
            range(train_start, steps_per_epoch),
            total=steps_per_epoch,
            initial=train_start,
            desc=f"Epoch {epoch + 1}/{epochs} [train]",
            unit="batch",
        )

        for _batch_idx in pbar:
            t0 = time.perf_counter()

            params_np = _sample_params_batch(rng_train, train_batch_size)
            target_np = _make_target_batch(params_np)

            params_batch = torch.from_numpy(params_np).float().to(DEVICE)
            target_batch = torch.from_numpy(target_np).float().to(DEVICE)

            forward_optimizer.zero_grad(set_to_none=True)

            pred_batch = predict_chi_batch_from_params(
                forward_model=forward_model,
                params_batch=params_batch,
                coords=coords,
                nx=Nx,
                ny=Ny,
                nz=Nz,
            )

            lhs = ApplyLUFn.apply(
                pred_batch,
                params_batch,
                Nx,
                Ny,
                Nz,
            )

            loss_f = forward_criterion(lhs, target_batch)

            t1 = time.perf_counter()
            loss_f.backward()
            t2 = time.perf_counter()
            forward_optimizer.step()
            t3 = time.perf_counter()

            train_loss_sum += loss_f.item()
            train_batches_done += 1
            avg_train = train_loss_sum / max(1, train_batches_done)

            pbar.set_postfix(
                loss=f"{loss_f.item():.3e}",
                avg=f"{avg_train:.3e}",
                pred=f"{t1 - t0:.2f}s",
                back=f"{t2 - t1:.2f}s",
                step=f"{t3 - t2:.2f}s",
                lr=f"{forward_optimizer.param_groups[0]['lr']:.1e}",
            )
        # -------------------------
        # VALIDATION PHASE
        # -------------------------
        forward_model.eval()

        pbar_val = tqdm(
            range(val_start, num_val_samples, val_batch_size),
            desc=f"Epoch {epoch + 1}/{epochs} [val]",
            unit="batch",
        )

        with torch.inference_mode():
            for batch_start in pbar_val:
                batch_end = min(batch_start + val_batch_size, num_val_samples)

                params_batch = val_params_all[batch_start:batch_end].to(
                    DEVICE, non_blocking=True
                )
                target_batch = val_target_all[batch_start:batch_end].to(
                    DEVICE, non_blocking=True
                )

                pred_batch = predict_chi_batch_from_params(
                    forward_model=forward_model,
                    params_batch=params_batch,
                    coords=coords,
                    nx=Nx,
                    ny=Ny,
                    nz=Nz,
                )

                lhs = ApplyLUFn.apply(
                    pred_batch,
                    params_batch,
                    Nx,
                    Ny,
                    Nz,
                )

                delta_val_loss = forward_criterion(lhs, target_batch).item()

                val_loss_sum += delta_val_loss
                val_batches_done += 1

                # Fixed correction: average over completed validation batches
                avg_val = val_loss_sum / max(1, val_batches_done)

                pbar_val.set_postfix(
                    loss=f"{delta_val_loss:.3e}",
                    avg=f"{avg_val:.3e}",
                )

        average_train_loss_f = train_loss_sum / max(1, train_batches_done)
        average_val_loss_f = val_loss_sum / max(
            1, (num_val_samples + val_batch_size - 1) // val_batch_size
        )

        loss_history["train_loss"].append(average_train_loss_f)
        loss_history["val_loss"].append(average_val_loss_f)

        # Save history every epoch
        save_loss_history(loss_history, loss_history_path)

        scheduler_f.step(average_val_loss_f)
        lr_f = forward_optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch + 1:03d}/{epochs} | "
            f"Fwd Loss (Tr/Val): {average_train_loss_f:.2e} / {average_val_loss_f:.2e} "
            f"[LR: {lr_f:.1e}]"
        )
        torch.save(forward_model.state_dict(), final_model_path)
        if average_val_loss_f < best_val_loss_f:
            best_val_loss_f = average_val_loss_f
            torch.save(forward_model.state_dict(), best_model_path)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Reset epoch accumulators for the next epoch
        train_loss_sum = 0.0
        train_batches_done = 0
        val_loss_sum = 0.0
        val_batches_done = 0

        if epochs_without_improvement >= patience:
            print("Early stopping triggered!")
            break

    print("Training complete.")

    with loss_history_path.open("w", encoding="utf-8") as f:
        json.dump(loss_history, f, indent=4)
    print(f"--> Saved metrics data to: {loss_history_path}")

    plot_training_convergence(loss_history, plot_path)

    torch.save(forward_model.state_dict(), final_model_path)
    print(f"--> Saved final model to: {final_model_path}")


def test() -> None:  # ruff: ignore[too-many-locals] # ruff: ignore[too-many-statements]
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
                np.array([3 * angstrom_si, 0, 0]),
                np.array([0, 3 * angstrom_si, 0]),
                np.array([0, 0, Z_HEIGHT * angstrom_si]),
            ),
            (9, 9, 100),
        ),
        incident_k=momentum_from_angles(
            theta=np.deg2rad(30),
            phi=np.deg2rad(0),
            energy=HELIUM_ENERGY,
            mass=HELIUM_MASS,
        ),
    )

    config = OptimizationConfig(precision=1e-5, max_iterations=1000, n_channels=250)

    test_params = params_from_condition(condition)

    nx, ny, nz = condition.metadata.shape
    coords = _make_coords(nx, ny, nz, device=DEVICE, dtype=torch.float32)

    basis = close_coupling_basis(condition.metadata)

    def _to_channel_array(state: State) -> np.ndarray:
        """Channel-space (Nx, Ny, Nz) array for a state in any basis."""
        return state.with_basis(basis).raw_data.reshape(condition.metadata.shape)

    def _save_s_matrix(s_matrix: Array, title: str, path: str) -> None:
        fig, ax, _mech = plot.array_against_axes_2d_k_nearest_neighbor(
            s_matrix, measure="abs"
        )
        ax.set_title(title)
        fig.savefig(path)

    # -------------------------
    # PREDICTION
    # -------------------------
    forward_model = PureMLP().to(DEVICE)
    forward_model.load_state_dict(
        torch.load(
            "data/state_model/chi_model.pth",
            map_location=DEVICE,
        ),
    )
    forward_model.eval()

    with torch.no_grad():
        param_tensor = torch.tensor(
            test_params,
            dtype=torch.float32,
            device=DEVICE,
        ).unsqueeze(0)  # (1, n_params)

        channel_amp_dense_batch = predict_chi_batch_from_params(
            forward_model=forward_model,
            params_batch=param_tensor,  # Only pass the physical parameters
            coords=coords,
            nx=nx,
            ny=ny,
            nz=nz,
        )

        channel_amp_dense = channel_amp_dense_batch[0]  # shape: (2, Nx, Ny, Nz)
        # 2. Combine channels into a complex tensor: Shape (Nx, Ny, Nz)
        state_data_complex = torch.complex(channel_amp_dense[0], channel_amp_dense[1])

        # 4. Transfer to NumPy for your physical State class
        preconditioned_pred_state = State(
            basis.upcast(),
            state_data_complex.detach().cpu().numpy(),
        )

    # -------------------------
    # GROUND TRUTH
    # -------------------------
    actual = get_scattering_state(
        condition,
        config,
    )
    preconditioned_state = get_preconditioned_state_from_state(
        actual, condition, n_channels=config.n_channels
    )

    pred_channels = _to_channel_array(preconditioned_pred_state)
    actual_channels = _to_channel_array(preconditioned_state)

    # -------------------------
    # PLOTS
    # -------------------------
    _, metadata_z = split_scattering_metadata(condition.metadata)
    z = metadata_z.values
    fig, ax1 = plot.get_figure()
    ax1.set_xlabel("z")
    ax1.set_ylabel(r"$\psi_{00}(z)$")
    ax1.plot(z, actual_channels[0, 0, :].real, label="Actual real part")
    ax1.plot(z, pred_channels[0, 0, :].real, label="Predicted real part")
    ax1.set_title("Actual and predicted scattering state")
    ax1.legend()
    fig.savefig("data/state_model/result_figures/scattering_state.png")

    actual_s_matrix = get_scattering_matrix_from_preconditioned_state(
        preconditioned_state, condition
    )
    predicted_s_matrix = get_scattering_matrix_from_preconditioned_state(
        preconditioned_pred_state,  # ty: ignore[invalid-argument-type]
        condition,
    )

    _save_s_matrix(
        actual_s_matrix,
        "The actual scattering matrix",
        "data/state_model/result_figures/scattering_matrix_from_actual_state.png",
    )
    _save_s_matrix(
        predicted_s_matrix,
        "The predicted scattering matrix",
        "data/state_model/result_figures/scattering_matrix_from_predicted_state.png",
    )

    # -------------------------
    # METRICS
    # -------------------------
    def _metrics(
        predicted: np.ndarray,
        actual: np.ndarray,
        name: str,
    ) -> None:
        if predicted.shape != actual.shape:
            msg = (
                f"{name} shape mismatch: "
                f"predicted {predicted.shape}, actual {actual.shape}."
            )
            raise ValueError(msg)

        difference = (predicted - actual).ravel()
        actual_flat = actual.ravel()

        rmse = float(np.linalg.norm(difference) / np.sqrt(difference.size))
        relative_l2 = float(np.linalg.norm(difference) / np.linalg.norm(actual_flat))
        cos_sim = float(
            np.abs(np.vdot(actual_flat, predicted.ravel()))
            / (np.linalg.norm(actual_flat) * np.linalg.norm(predicted.ravel()))
        )
        r_squared = float(
            1.0
            - np.sum(np.abs(difference) ** 2)
            / np.sum(np.abs(actual_flat - actual_flat.mean()) ** 2)
        )

        print(f"\n[{name}]")
        print(f"RMSE   : {rmse:.6e}")
        print(f"RelL2  : {relative_l2:.6e}")
        print(f"CosSim : {cos_sim:.6f}")
        print(f"R^2    : {r_squared:.6f}")

    _metrics(pred_channels, actual_channels, name="Channel space")
    _metrics(
        predicted_s_matrix.raw_data, actual_s_matrix.raw_data, name="Scattering matrix"
    )

    with Path("data/state_model/loss_history.json").open("r", encoding="utf-8") as f:
        loss_history = json.load(f)

    plot_training_convergence(
        loss_history, Path("data/state_model/convergence_curve.png")
    )


def test_preconditioned_identity() -> None:  # ruff: ignore[too-many-locals]
    """Check (I + L^-1 U) psi = L^-1 b for the exact solver solution."""
    params = np.array([0.5, 0.5, 0.25, 0.5, 0.5, 1 / 3], dtype=np.float64)

    condition = _as_natural_units(condition_from_params(params))
    condition0 = condition_from_params(params)
    basis0 = close_coupling_basis(condition0.metadata)
    config = OptimizationConfig(precision=1e-8, max_iterations=1000, n_channels=250)

    inverse_lower, _lower, upper = _build_scipy_operators(condition, n_channels=250)
    basis = close_coupling_basis(condition.metadata)

    # RHS: L^-1 b, exactly as simulate_target_state builds the training target
    target = get_target_state(condition.metadata, condition.incident_k)
    rhs = inverse_lower.matvec(target.with_basis(basis).raw_data)

    # LHS candidates: which state does the identity actually hold for?
    exact = get_scattering_state(condition0, config)
    preconditioned = get_preconditioned_state_from_state(
        exact, condition0, n_channels=config.n_channels
    )

    for name, state in (("psi", exact), ("preconditioned psi", preconditioned)):
        chi = state.with_basis(basis0).raw_data
        lhs = chi + inverse_lower.matvec(upper.matvec(chi))

        residual = np.linalg.norm(lhs - rhs) / np.linalg.norm(rhs)
        print(f"{name:<20}: relative residual = {residual:.3e}")


def test_adjoint(_nx: int = 9, _ny: int = 9, _nz: int = 100) -> None:  # ruff: ignore[pytest-fixture-param-without-value]  # ruff: ignore[pytest-parameter-with-default-argument]
    """Check ApplyLUFn's backward is the adjoint of its forward.

    For a linear map, <A x, y> == <x, J^T y> exactly, where J^T y is what
    backward returns. One forward and one backward, no finite differences.
    """
    params_phys = np.array(
        [7.63, 1.0 / 1.1, 1.0, 0.05, 2.84, np.pi / 2], dtype=np.float64
    )
    params = torch.as_tensor(
        normalize_params(params_phys), dtype=torch.float64
    ).unsqueeze(0)

    gen = torch.Generator().manual_seed(0)
    x = torch.randn(
        1, 2, _nx, _ny, _nz, generator=gen, dtype=torch.float64, requires_grad=True
    )
    y = torch.randn(1, 2, _nx, _ny, _nz, generator=gen, dtype=torch.float64)

    ax = ApplyLUFn.apply(x, params, _nx, _ny, _nz)
    lhs = torch.sum(ax * y)
    (jty,) = torch.autograd.grad(lhs, x)
    rhs = torch.sum(x.detach() * jty)

    rel = abs(float(lhs) - float(rhs)) / max(abs(float(lhs)), 1e-30)
    print(f"<Ax, y> = {float(lhs): .12e}")
    print(f"<x, A^H y> = {float(rhs): .12e}")
    print(f"relative error = {rel:.2e}   {'PASS' if rel < 1e-9 else 'FAIL'}")  # ruff: ignore[magic-value-comparison]


if __name__ == "__main__":
    RUN_TRAIN_RAND = False
    RUN_TEST = True

    if RUN_TRAIN_RAND:
        train_rand()
    if RUN_TEST:
        test()
