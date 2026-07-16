import contextlib
import json
import time
from pathlib import Path
from typing import Any, override

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.constants import (  # type: ignore[import-untyped]
    electron_volt,
    physical_constants,
)
from slate_core.metadata import LobattoSpacedLengthMetadata
from torch import nn, optim
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
    [-3, -4, -5],
    dtype=np.float64,
)
PARAMS_MAX = np.array(
    [3, 4, 5],
    dtype=np.float64,
)

x = torch.linspace(-10, 10, 200)
y = torch.linspace(-10, 10, 200)
z = torch.linspace(-10, 10, 200)


def denormalize_params(
    params_norm: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    """Scales parameters back to their original physical units."""
    return params_norm * (PARAMS_MAX - PARAMS_MIN) + PARAMS_MIN


def normalize_params(
    params: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    """Scales parameters to a [0, 1] range."""
    return (params - PARAMS_MIN) / (PARAMS_MAX - PARAMS_MIN)


def simulate_function(
    params: np.ndarray,
    x_grid,
    y_grid,
    z_grid,
) -> np.ndarray:
    """
    Simple real-valued test function.

    Parameters
    ----------
    params
        Normalized parameters in [0, 1], shape (3,).

    Returns
    -------
    field : ndarray of shape (Nx, Ny, Nz)
    """
    if x_grid is None:
        x_grid = x
    if y_grid is None:
        y_grid = y
    if z_grid is None:
        z_grid = z

    kx, ky, kz = denormalize_params(params)

    X, Y, Z = torch.meshgrid(x_grid, y_grid, z_grid, indexing="ij")

    field = torch.sin(kx * X) * torch.sin(ky * Y) * torch.sin(kz * Z)

    return field.cpu().numpy()


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


class SirenLayer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        is_first=False,
        omega_0=5.0,
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
        omega_0: float = 5.0,
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
        param_dim: int = 3,  # kx, ky, kz
        coord_dim: int = 3,  # x, y, z
        cond_dim: int = 64,
        hidden_dim: int = 64,
        output_dim: int = 1,
        num_siren_layers: int = 4,
        first_omega_0: float = 5.0,
        hidden_omega_0: float = 5.0,
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
        params: (B, 3)
        coords: (N_pts, 3)
        returns: (B, N_pts, 1).
        """
        B = params.shape[0]
        N = coords.shape[0]

        cond = self.condition_encoder(params)  # (B, cond_dim)
        cond = cond[:, None, :].expand(B, N, -1)  # (B, N, cond_dim)

        x = coords[None, :, :].expand(B, N, -1)  # (B, N, 3)

        x = x.reshape(B * N, 3)
        cond = cond.reshape(B * N, -1)

        for layer in self.siren_layers:
            x = layer(x, cond)

        out = self.head(x)  # (B*N, 1)
        return out.view(B, N, 1)


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


def _sample_params_batch(rng: np.random.Generator, batch_size: int) -> np.ndarray:
    return rng.uniform(size=(batch_size, 3))


def _make_target_batch(
    params_batch_np: np.ndarray,
    x_grid: torch.Tensor = x,
    y_grid: torch.Tensor = y,
    z_grid: torch.Tensor = z,
) -> np.ndarray:
    return np.stack(
        [
            simulate_function(params, x_grid, y_grid, z_grid)
            for params in params_batch_np
        ],
        axis=0,
    )


def _make_fixed_val_set(
    rng: np.random.Generator,
    num_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    params_np = _sample_params_batch(rng, num_samples)
    target_np = _make_target_batch(params_np)

    params = torch.from_numpy(params_np).float()
    target = torch.from_numpy(target_np).float()
    return params, target


def _make_pred_batch(forward_model, params_batch_np, coords):
    """
    params_batch_np: (B, param_dim) numpy array
    coords: (N_pts, 3) flattened coordinates or a torch tensor.

    Returns
    -------
        preds_np: (B, Nx, Ny, Nz)
    """
    forward_model.eval()

    device = next(forward_model.parameters()).device
    params_batch = torch.as_tensor(params_batch_np, dtype=torch.float32, device=device)
    coords_t = torch.as_tensor(coords, dtype=torch.float32, device=device)

    preds = []
    with torch.no_grad():
        for i in range(params_batch.shape[0]):
            pred = forward_model(params_batch[i].unsqueeze(0), coords_t)

            # Expected shapes:
            #   (1, N_pts, 1) or (N_pts, 1)
            if pred.ndim == 3 and pred.shape[0] == 1:
                pred = pred.squeeze(0)  # (N_pts, 1)

            if pred.ndim != 2 or pred.shape[-1] != 1:
                msg = f"Expected shape (N_pts, 1), got {pred.shape}"
                raise ValueError(msg)

            pred = pred.squeeze(-1)  # (N_pts,)
            preds.append(pred)

    preds = torch.stack(preds, dim=0)  # (B, N_pts)

    grid_size = len(x) * len(y) * len(z)
    if coords_t.shape[0] == grid_size:
        preds = preds.view(params_batch.shape[0], len(x), len(y), len(z))

    return preds.cpu().numpy()


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


def train() -> None:
    forward_model = ForwardCondSIRENStateModel(
        param_dim=3,
        output_dim=1,
    ).to(DEVICE)

    forward_criterion = WavePhysicsLoss()
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
    output_dir = Path("data/15")
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss_f = float("inf")
    patience = 150
    epochs_without_improvement = 0
    epochs = 150

    print(f"Using device: {DEVICE}")

    train_batch_size = 64
    steps_per_epoch = 100
    num_val_samples = 512

    rng_train = np.random.default_rng()
    rng_val = np.random.default_rng(12345)

    # Build the fixed coordinate grid once
    X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")
    coords = (
        torch.stack(
            [X.reshape(-1), Y.reshape(-1), Z.reshape(-1)],
            dim=-1,
        )
        .to(DEVICE)
        .float()
    )

    Nx, Ny, Nz = len(x), len(y), len(z)

    # Fixed validation set
    val_params_all, val_target_all = _make_fixed_val_set(rng_val, num_val_samples)
    val_params_all = val_params_all.to(DEVICE)
    val_target_all = val_target_all.to(DEVICE)

    for epoch in range(epochs):
        forward_model.train()
        train_loss_f = 0.0

        pbar = tqdm(
            range(steps_per_epoch), desc=f"Epoch {epoch + 1}/{epochs}", unit="batch"
        )

        for _batch_idx in pbar:
            t0 = time.perf_counter()

            params_np = _sample_params_batch(rng_train, train_batch_size)
            target_np = _make_target_batch(params_np)

            params_batch = torch.from_numpy(params_np).float().to(DEVICE)
            target_batch = torch.from_numpy(target_np).float().to(DEVICE)

            forward_optimizer.zero_grad(set_to_none=True)

            # Training forward pass: keep gradients on
            pred_flat = forward_model(params_batch, coords)  # (B, N_pts, 1)
            pred_batch = (
                pred_flat[..., 0].contiguous().view(train_batch_size, Nx, Ny, Nz)
            )  # (B, Nx, Ny, Nz)

            loss_f = forward_criterion(pred_batch, target_batch)
            loss_f.backward()

            torch.nn.utils.clip_grad_norm_(forward_model.parameters(), max_norm=1.0)
            forward_optimizer.step()

            train_loss_f += loss_f.item()

            pbar.set_postfix(
                loss=f"{loss_f.item():.4e}",
                ms=f"{(time.perf_counter() - t0) * 1000:.1f}",
            )

        train_loss_f /= steps_per_epoch
        loss_history["train_loss"].append(train_loss_f)

        # Validation
        forward_model.eval()
        with torch.no_grad():
            val_pred_flat = forward_model(val_params_all, coords)  # (V, N_pts, 1)
            val_pred = (
                val_pred_flat[..., 0].contiguous().view(num_val_samples, Nx, Ny, Nz)
            )
            val_loss_f = forward_criterion(val_pred, val_target_all).item()

        loss_history["val_loss"].append(val_loss_f)
        scheduler_f.step(val_loss_f)

        print(
            f"Epoch {epoch + 1:03d} | "
            f"train_loss={train_loss_f:.4e} | "
            f"val_loss={val_loss_f:.4e}"
        )

        if val_loss_f < best_val_loss_f:
            best_val_loss_f = val_loss_f
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": forward_model.state_dict(),
                    "optimizer_state_dict": forward_optimizer.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss_f,
                },
                output_dir / "best_test_forward_model.pt",
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    print("Training complete.")

    with Path(output_dir / "loss_history.json").open("w", encoding="utf-8") as f:
        json.dump(loss_history, f, indent=4)
    print(f"--> Saved metrics data to: {output_dir / 'loss_history.json'}")

    plot_training_convergence(loss_history, output_dir / "convergence_curve_test.png")

    torch.save(forward_model.state_dict(), "data/15/test_model.pth")


def test() -> None:
    """
    Compare actual vs predicted real-valued field on a fresh grid.

    ```
    Expected model contract:
        forward_model(params, coords) -> (B, N_pts, 1)

    Expected target:
        field shape (Nx, Ny, Nz)
    """
    # ------------------------------------------------------------------
    # 1. Choose a test condition / parameter set
    # ------------------------------------------------------------------
    # params should be normalized if your training used normalized inputs.
    # Example: params_norm = np.array([0.3, 0.6, 0.8], dtype=np.float64)
    params_norm = np.array([0.3, 0.6, 0.8], dtype=np.float64)

    # If you already have physical parameters, skip denormalize_params
    # and adapt simulate_function_on_grid accordingly.

    # ------------------------------------------------------------------
    # 2. Build a fresh test grid
    # ------------------------------------------------------------------
    Nx, Ny, Nz = 160, 160, 200
    x_grid = np.linspace(-10.0, 10.0, Nx, dtype=np.float32)
    y_grid = np.linspace(-10.0, 10.0, Ny, dtype=np.float32)
    z_grid = np.linspace(-10.0, 10.0, Nz, dtype=np.float32)

    X, Y, Z = torch.meshgrid(
        torch.from_numpy(x_grid),
        torch.from_numpy(y_grid),
        torch.from_numpy(z_grid),
        indexing="ij",
    )

    coords = (
        torch.stack(
            (
                X.reshape(-1),
                Y.reshape(-1),
                Z.reshape(-1),
            ),
            dim=-1,
        )
        .to(DEVICE)
        .float()
    )

    # ------------------------------------------------------------------
    # 3. Load the model
    # ------------------------------------------------------------------
    forward_model = ForwardCondSIRENStateModel(
        param_dim=3,
        output_dim=1,
    ).to(DEVICE)

    forward_model.load_state_dict(
        torch.load("data/15/best_test_forward_model.pt", map_location=DEVICE)[
            "model_state_dict"
        ]
    )
    forward_model.eval()

    # ------------------------------------------------------------------
    # 4. Ground truth and prediction
    # ------------------------------------------------------------------
    actual_field = simulate_function(params_norm, x_grid, y_grid, z_grid)

    with torch.no_grad():
        param_tensor = torch.tensor(
            params_norm,
            dtype=torch.float32,
            device=DEVICE,
        ).unsqueeze(0)  # (1, 3)

        pred_field_batch = forward_model(param_tensor, coords)  # (1, N_pts, 1)

        if pred_field_batch.ndim != 3 or pred_field_batch.shape[-1] != 1:
            msg = f"Unexpected model output shape: {pred_field_batch.shape}"
            raise ValueError(msg)

        pred_field = pred_field_batch[0, :, 0].reshape(Nx, Ny, Nz).cpu().numpy()

    # ------------------------------------------------------------------
    # 5. Pick slice locations
    # ------------------------------------------------------------------
    center_x = Nx // 2
    center_y = Ny // 2
    center_z = Nz // 2

    # 1D slice: f(z) at fixed x, y
    actual_z = actual_field[center_x, center_y, :]
    pred_z = pred_field[center_x, center_y, :]

    # 2D slice: f(x, y) at fixed z
    actual_xy = actual_field[:, :, center_z]
    pred_xy = pred_field[:, :, center_z]

    # ------------------------------------------------------------------
    # 6. Plot 1D and 2D comparisons
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    ax.plot(z_grid, actual_z, label="Actual", linewidth=2)
    ax.plot(z_grid, pred_z, "--", label="Predicted", linewidth=2)
    ax.set_xlabel("z")
    ax.set_ylabel("f(x0, y0, z)")
    ax.set_title(f"1D slice at x={x_grid[center_x]:.3f}, y={y_grid[center_y]:.3f}")
    ax.legend(frameon=True)

    ax = axes[0, 1]
    ax.plot(z_grid, np.abs(pred_z - actual_z))
    ax.set_xlabel("z")
    ax.set_ylabel("|error|")
    ax.set_title("Absolute error along z")

    extent = [x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]]

    ax = axes[1, 0]
    im0 = ax.imshow(
        actual_xy.T,
        origin="lower",
        aspect="auto",
        extent=extent,
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Actual x-y slice at z={z_grid[center_z]:.3f}")
    fig.colorbar(im0, ax=ax, shrink=0.85)

    ax = axes[1, 1]
    im1 = ax.imshow(
        pred_xy.T,
        origin="lower",
        aspect="auto",
        extent=extent,
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Predicted x-y slice at z={z_grid[center_z]:.3f}")
    fig.colorbar(im1, ax=ax, shrink=0.85)

    plt.tight_layout()
    fig.savefig(
        "data/15/simple_field_prediction_test.png", bbox_inches="tight", dpi=300
    )

    print("--> Simple field comparison plot saved successfully.")


if __name__ == "__main__":
    RUN_TRAIN = True
    RUN_TEST = True

    if RUN_TRAIN:
        train()
    if RUN_TEST:
        test()
