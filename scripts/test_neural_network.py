import contextlib
import json
import time
from pathlib import Path
from typing import Any, override

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from tqdm import tqdm

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")  # pyright: ignore[reportConstantRedefinition]
else:
    DEVICE = torch.device("cpu")  # pyright: ignore[reportConstantRedefinition]

PARAMS_MIN = np.array(
    [-3, -4, -5, -6, 0, 3],
    dtype=np.float64,
)
PARAMS_MAX = np.array(
    [3, 4, 5, 6, 25, 8],
    dtype=np.float64,
)

x = torch.linspace(-1, 1, 20)
y = torch.linspace(-1, 1, 20)
z = torch.linspace(-1, 1, 20)


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

    kx, ky, kz, k1, k2, k3 = denormalize_params(params)

    X, Y, Z = torch.meshgrid(x_grid, y_grid, z_grid, indexing="ij")

    field = (
        torch.sin(kx * k1**2 * X) * torch.sin(ky * k2**0.5 * Y) * torch.sin(kz * Z / k3)
    )

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
        omega_0=10.0,
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
            nn.Linear(128, 128),
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
        param_dim: int = 6,  # kx, ky, kz, k1, k2, k3
        coord_dim: int = 3,  # x, y, z
        cond_dim: int = 64,
        hidden_dim: int = 64,
        output_dim: int = 1,
        num_siren_layers: int = 8,
        first_omega_0: float = 10.0,
        hidden_omega_0: float = 10.0,
    ) -> None:
        super().__init__()

        self.hidden_omega_0 = hidden_omega_0

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
        self.init_head()

    def init_head(self) -> None:
        with torch.no_grad():
            bound = np.sqrt(6.0 / self.head.in_features) / self.hidden_omega_0
            self.head.weight.uniform_(-bound, bound)
            if self.head.bias is not None:
                self.head.bias.zero_()

    def forward(self, params: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        B = params.shape[0]
        N = coords.shape[0]

        cond = self.condition_encoder(params)  # (B, cond_dim)
        cond = cond[:, None, :].expand(B, N, -1)  # (B, N, cond_dim)

        x = coords[None, :, :].expand(B, N, -1)  # (B, N, 3)

        x = x.reshape(B * N, 3)
        cond = cond.reshape(B * N, -1)

        for layer in self.siren_layers:
            x = layer(x, cond)

        out = self.head(x)
        return out.view(B, N, 1)


class PureSIREN(nn.Module):
    """
    Input:
        params: (B, 3)   -> normalized kx, ky, kz, k1, k2, k3 in [0, 1] or standardized
        coords: (N, 3)   -> x, y, z coordinates.

    Output:
        (B, N, 1)
    """

    def __init__(
        self,
        param_dim: int = 6,
        coord_dim: int = 3,
        hidden_dim: int = 512,
        num_layers: int = 4,
        first_omega_0: float = 10.0,
        hidden_omega_0: float = 10.0,
        output_dim: int = 1,
    ) -> None:
        super().__init__()

        self.in_dim = param_dim + coord_dim
        self.hidden_omega_0 = hidden_omega_0

        layers = [
            SirenLayer(
                in_features=self.in_dim,
                out_features=hidden_dim,
                is_first=True,
                omega_0=first_omega_0,
            )
        ]

        layers.extend(
            SirenLayer(
                in_features=hidden_dim,
                out_features=hidden_dim,
                is_first=False,
                omega_0=hidden_omega_0,
            )
            for _ in range(num_layers - 1)
        )

        self.net = nn.ModuleList(layers)
        self.head = nn.Linear(hidden_dim, output_dim)
        self.init_head()

    def init_head(self) -> None:
        with torch.no_grad():
            bound = np.sqrt(6.0 / self.head.in_features) / self.hidden_omega_0
            self.head.weight.uniform_(-bound, bound)
            if self.head.bias is not None:
                self.head.bias.zero_()

    def forward(self, params: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        params: (B, 3)
        coords: (N, 3)
        returns: (B, N, 1).
        """
        B = params.shape[0]
        N = coords.shape[0]

        params_expanded = params[:, None, :].expand(B, N, -1)  # (B, N, 3)
        coords_expanded = coords[None, :, :].expand(B, N, -1)  # (B, N, 3)

        x = torch.cat([params_expanded, coords_expanded], dim=-1)  # (B, N, 6)
        x = x.reshape(B * N, -1)

        for layer in self.net:
            x = layer(x)

        out = self.head(x)
        return out.view(B, N, 1)


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
    """
    Plain MLP baseline with the same interface as PureSIREN.

    Input:
        params : (B, param_dim)
        coords : (N, coord_dim)

    Output:
        (B, N, 1)
    """

    def __init__(
        self,
        param_dim: int = 6,
        coord_dim: int = 3,
        hidden_dim: int = 512,
        num_blocks: int = 4,
        output_dim: int = 1,
    ) -> None:
        super().__init__()

        self.output_dim = output_dim

        self.embedding = nn.Sequential(
            nn.Linear(param_dim + coord_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.res_blocks = nn.Sequential(
            *[ResBlock(hidden_dim) for _ in range(num_blocks)]
        )

        self.head = nn.Linear(hidden_dim, output_dim)

        self.init_weights()

    def init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        params: torch.Tensor,
        coords: torch.Tensor,
    ) -> torch.Tensor:

        B = params.shape[0]
        N = coords.shape[0]

        params = params[:, None, :].expand(B, N, -1)
        coords = coords[None, :, :].expand(B, N, -1)

        x = torch.cat([params, coords], dim=-1)
        x = x.reshape(B * N, -1)

        x = self.embedding(x)
        x = self.res_blocks(x)
        x = self.head(x)

        return x.view(B, N, self.output_dim)


class WavePhysicsLoss(nn.Module):
    """
    Loss for scalar fields.

    Components
    ----------
    Relative L2:
        ||u_pred - u|| / ||u||

    Cosine similarity:
        Encourages the predicted field to have the same global structure.

    Amplitude loss (optional):
        Wasserstein-style comparison of the value distributions.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps
        self.cosine = nn.CosineSimilarity(dim=1)

    def forward(self, pred, target, alpha: float = 0.0):
        # pred,target: (B,Nx,Ny,Nz)

        B = pred.shape[0]

        pred = pred.reshape(B, -1)
        target = target.reshape(B, -1)

        # ----------------------------------------------------
        # Relative L2
        # ----------------------------------------------------
        diff_norm = torch.norm(pred - target, p=2, dim=1)
        target_norm = torch.norm(target, p=2, dim=1)

        rel_l2 = (diff_norm / (target_norm + self.eps)).mean()

        # ----------------------------------------------------
        # Global structural similarity
        # ----------------------------------------------------
        cosine_loss = (1.0 - self.cosine(pred, target)).mean()

        if alpha == 0.0:
            return rel_l2 + cosine_loss

        # ----------------------------------------------------
        # Distribution matching (Wasserstein approximation)
        # ----------------------------------------------------
        pred_sorted = torch.sort(torch.abs(pred), dim=1).values
        target_sorted = torch.sort(torch.abs(target), dim=1).values

        amplitude_loss = torch.mean(torch.abs(pred_sorted - target_sorted))

        return rel_l2 + (1.0 - 0.5 * alpha) * cosine_loss + alpha * amplitude_loss


def _sample_params_batch(rng: np.random.Generator, batch_size: int) -> np.ndarray:
    return rng.uniform(size=(batch_size, 6))


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


def train_PureSIREN() -> None:
    forward_model = PureSIREN(
        param_dim=6,
        output_dim=1,
    ).to(DEVICE)

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

    loss_history = {"train_loss": [], "val_loss": []}
    output_dir = Path("data/15")
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss_f = float("inf")
    patience = 300
    epochs_without_improvement = 0
    epochs = 300

    print(f"Using device: {DEVICE}")

    train_batch_size = 16
    steps_per_epoch = 100
    num_val_samples = 32

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
        epoch_start = time.perf_counter()
        forward_model.train()
        train_loss_f = 0.0

        current_lr = forward_optimizer.param_groups[0]["lr"]
        print(
            f"\n{'=' * 90}\n"
            f"Epoch {epoch + 1:3d}/{epochs} | lr={current_lr:.2e} | "
            f"best_val={best_val_loss_f:.6e} | patience={epochs_without_improvement}/{patience}\n"
            f"{'=' * 90}"
        )

        pbar = tqdm(
            range(steps_per_epoch),
            desc=f"Epoch {epoch + 1}/{epochs}",
            unit="batch",
        )

        for batch_idx in pbar:
            t0 = time.perf_counter()

            params_np = _sample_params_batch(rng_train, train_batch_size)
            target_np = _make_target_batch(params_np)

            params_batch = torch.from_numpy(params_np).float().to(DEVICE)
            target_batch = torch.from_numpy(target_np).float().to(DEVICE)

            forward_optimizer.zero_grad(set_to_none=True)

            pred_flat = forward_model(params_batch, coords)  # (B, N_pts, 1)
            pred_batch = (
                pred_flat[..., 0].contiguous().view(train_batch_size, Nx, Ny, Nz)
            )

            loss_f = forward_criterion(pred_batch, target_batch)
            loss_f.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                forward_model.parameters(),
                max_norm=1.0,
            )
            forward_optimizer.step()

            train_loss_f += loss_f.item()
            running_loss = train_loss_f / (batch_idx + 1)

            if batch_idx % 10 == 0:
                print(
                    f"  batch {batch_idx:03d}/{steps_per_epoch} | "
                    f"loss={loss_f.item():.4e} | "
                    f"avg={running_loss:.4e} | "
                    f"pred_mean={pred_batch.mean().item():.3e} | "
                    f"pred_std={pred_batch.std().item():.3e} | "
                    f"tgt_std={target_batch.std().item():.3e} | "
                    f"grad_norm={float(grad_norm):.3e} | "
                    f"step_time={(time.perf_counter() - t0) * 1000:.1f} ms"
                )

            pbar.set_postfix(
                loss=f"{loss_f.item():.3e}",
                avg=f"{running_loss:.3e}",
                lr=f"{current_lr:.1e}",
            )

        train_loss_f /= steps_per_epoch
        loss_history["train_loss"].append(train_loss_f)

        forward_model.eval()
        with torch.no_grad():
            val_pred_flat = forward_model(val_params_all, coords)  # (V, N_pts, 1)
            val_pred = (
                val_pred_flat[..., 0].contiguous().view(num_val_samples, Nx, Ny, Nz)
            )
            val_loss_f = forward_criterion(val_pred, val_target_all).item()

        loss_history["val_loss"].append(val_loss_f)
        scheduler_f.step(val_loss_f)

        new_lr = forward_optimizer.param_groups[0]["lr"]
        epoch_time = time.perf_counter() - epoch_start

        print(
            f"Epoch {epoch + 1:03d} done | "
            f"train_loss={train_loss_f:.6e} | "
            f"val_loss={val_loss_f:.6e} | "
            f"best_val={best_val_loss_f:.6e} | "
            f"epoch_time={epoch_time:.2f} s"
        )

        if new_lr != current_lr:
            print(f"Learning rate reduced: {current_lr:.2e} -> {new_lr:.2e}")

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
                output_dir / "best_test_forward_model_Pure_SIREN.pth",
            )
            print("✓ New best model saved.")
        else:
            epochs_without_improvement += 1
            print(
                f"No improvement: {epochs_without_improvement}/{patience} "
                f"epochs without progress"
            )

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    print("Training complete.")

    with Path(output_dir / "loss_history.json").open("w", encoding="utf-8") as f:
        json.dump(loss_history, f, indent=4)
    print(f"--> Saved metrics data to: {output_dir / 'loss_history_PureSIREN.json'}")

    plot_training_convergence(
        loss_history, output_dir / "convergence_curve_test_PureSIREN.png"
    )

    torch.save(forward_model.state_dict(), "data/15/test_model_PureSIREN.pth")


def train_CondSIREN() -> None:
    forward_model = ForwardCondSIRENStateModel(
        param_dim=6,
        output_dim=1,
    ).to(DEVICE)

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

    loss_history = {"train_loss": [], "val_loss": []}
    output_dir = Path("data/15")
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss_f = float("inf")
    patience = 300
    epochs_without_improvement = 0
    epochs = 300

    switch_epoch = 40  # Start transitioning at epoch 40
    ramp_duration = 30  # Linearly blend the amplitude loss over 30 epochs

    print(f"Using device: {DEVICE}")

    train_batch_size = 16
    steps_per_epoch = 100
    num_val_samples = 32

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
        epoch_start = time.perf_counter()
        forward_model.train()
        # Calculate dynamic alpha for stage routing
        if epoch < switch_epoch:
            pass
        else:
            # Smoothly scales alpha from 0.0 to 1.0 to prevent optimizer shock
            min(1.0, (epoch - switch_epoch) / ramp_duration)
        train_loss_f = 0.0

        current_lr = forward_optimizer.param_groups[0]["lr"]
        print(
            f"\n{'=' * 90}\n"
            f"Epoch {epoch + 1:3d}/{epochs} | lr={current_lr:.2e} | "
            f"best_val={best_val_loss_f:.6e} | patience={epochs_without_improvement}/{patience}\n"
            f"{'=' * 90}"
        )

        pbar = tqdm(
            range(steps_per_epoch),
            desc=f"Epoch {epoch + 1}/{epochs}",
            unit="batch",
        )

        for batch_idx in pbar:
            t0 = time.perf_counter()

            params_np = _sample_params_batch(rng_train, train_batch_size)
            target_np = _make_target_batch(params_np)

            params_batch = torch.from_numpy(params_np).float().to(DEVICE)
            target_batch = torch.from_numpy(target_np).float().to(DEVICE)

            forward_optimizer.zero_grad(set_to_none=True)

            pred_flat = forward_model(params_batch, coords)  # (B, N_pts, 1)
            pred_batch = (
                pred_flat[..., 0].contiguous().view(train_batch_size, Nx, Ny, Nz)
            )

            loss_f = forward_criterion(pred_batch, target_batch)
            loss_f.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                forward_model.parameters(),
                max_norm=1.0,
            )
            forward_optimizer.step()

            train_loss_f += loss_f.item()
            running_loss = train_loss_f / (batch_idx + 1)

            if batch_idx % 10 == 0:
                print(
                    f"  batch {batch_idx:03d}/{steps_per_epoch} | "
                    f"loss={loss_f.item():.4e} | "
                    f"avg={running_loss:.4e} | "
                    f"pred_mean={pred_batch.mean().item():.3e} | "
                    f"pred_std={pred_batch.std().item():.3e} | "
                    f"tgt_std={target_batch.std().item():.3e} | "
                    f"grad_norm={float(grad_norm):.3e} | "
                    f"step_time={(time.perf_counter() - t0) * 1000:.1f} ms"
                )

            pbar.set_postfix(
                loss=f"{loss_f.item():.3e}",
                avg=f"{running_loss:.3e}",
                lr=f"{current_lr:.1e}",
            )

        train_loss_f /= steps_per_epoch
        loss_history["train_loss"].append(train_loss_f)

        forward_model.eval()
        with torch.no_grad():
            val_pred_flat = forward_model(val_params_all, coords)  # (V, N_pts, 1)
            val_pred = (
                val_pred_flat[..., 0].contiguous().view(num_val_samples, Nx, Ny, Nz)
            )
            val_loss_f = forward_criterion(val_pred, val_target_all).item()

        loss_history["val_loss"].append(val_loss_f)
        scheduler_f.step(val_loss_f)

        new_lr = forward_optimizer.param_groups[0]["lr"]
        epoch_time = time.perf_counter() - epoch_start

        print(
            f"Epoch {epoch + 1:03d} done | "
            f"train_loss={train_loss_f:.6e} | "
            f"val_loss={val_loss_f:.6e} | "
            f"best_val={best_val_loss_f:.6e} | "
            f"epoch_time={epoch_time:.2f} s"
        )

        if new_lr != current_lr:
            print(f"Learning rate reduced: {current_lr:.2e} -> {new_lr:.2e}")

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
                output_dir / "best_test_forward_model_CondSIREN.pth",
            )
            print("✓ New best model saved.")
        else:
            epochs_without_improvement += 1
            print(
                f"No improvement: {epochs_without_improvement}/{patience} "
                f"epochs without progress"
            )

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    print("Training complete.")

    with Path(output_dir / "loss_history.json").open("w", encoding="utf-8") as f:
        json.dump(loss_history, f, indent=4)
    print(f"--> Saved metrics data to: {output_dir / 'loss_history_CondSIREN.json'}")

    plot_training_convergence(
        loss_history, output_dir / "convergence_curve_test_CondSIREN.png"
    )

    torch.save(forward_model.state_dict(), "data/15/test_model_CondSIREN.pth")


def train_PlainML() -> None:
    forward_model = PureMLP(
        param_dim=6,
        output_dim=1,
    ).to(DEVICE)

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

    loss_history = {"train_loss": [], "val_loss": []}
    output_dir = Path("data/15")
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss_f = float("inf")
    patience = 300
    epochs_without_improvement = 0
    epochs = 300

    switch_epoch = 40  # Start transitioning at epoch 40
    ramp_duration = 30  # Linearly blend the amplitude loss over 30 epochs

    print(f"Using device: {DEVICE}")

    train_batch_size = 16
    steps_per_epoch = 100
    num_val_samples = 32

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
        epoch_start = time.perf_counter()
        forward_model.train()
        # Calculate dynamic alpha for stage routing
        if epoch < switch_epoch:
            pass
        else:
            # Smoothly scales alpha from 0.0 to 1.0 to prevent optimizer shock
            min(1.0, (epoch - switch_epoch) / ramp_duration)
        train_loss_f = 0.0

        current_lr = forward_optimizer.param_groups[0]["lr"]
        print(
            f"\n{'=' * 90}\n"
            f"Epoch {epoch + 1:3d}/{epochs} | lr={current_lr:.2e} | "
            f"best_val={best_val_loss_f:.6e} | patience={epochs_without_improvement}/{patience}\n"
            f"{'=' * 90}"
        )

        pbar = tqdm(
            range(steps_per_epoch),
            desc=f"Epoch {epoch + 1}/{epochs}",
            unit="batch",
        )

        for batch_idx in pbar:
            t0 = time.perf_counter()

            params_np = _sample_params_batch(rng_train, train_batch_size)
            target_np = _make_target_batch(params_np)

            params_batch = torch.from_numpy(params_np).float().to(DEVICE)
            target_batch = torch.from_numpy(target_np).float().to(DEVICE)

            forward_optimizer.zero_grad(set_to_none=True)

            pred_flat = forward_model(params_batch, coords)  # (B, N_pts, 1)
            pred_batch = (
                pred_flat[..., 0].contiguous().view(train_batch_size, Nx, Ny, Nz)
            )

            loss_f = forward_criterion(pred_batch, target_batch)
            loss_f.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                forward_model.parameters(),
                max_norm=1.0,
            )
            forward_optimizer.step()

            train_loss_f += loss_f.item()
            running_loss = train_loss_f / (batch_idx + 1)

            if batch_idx % 10 == 0:
                print(
                    f"  batch {batch_idx:03d}/{steps_per_epoch} | "
                    f"loss={loss_f.item():.4e} | "
                    f"avg={running_loss:.4e} | "
                    f"pred_mean={pred_batch.mean().item():.3e} | "
                    f"pred_std={pred_batch.std().item():.3e} | "
                    f"tgt_std={target_batch.std().item():.3e} | "
                    f"grad_norm={float(grad_norm):.3e} | "
                    f"step_time={(time.perf_counter() - t0) * 1000:.1f} ms"
                )

            pbar.set_postfix(
                loss=f"{loss_f.item():.3e}",
                avg=f"{running_loss:.3e}",
                lr=f"{current_lr:.1e}",
            )

        train_loss_f /= steps_per_epoch
        loss_history["train_loss"].append(train_loss_f)

        forward_model.eval()
        with torch.no_grad():
            val_pred_flat = forward_model(val_params_all, coords)  # (V, N_pts, 1)
            val_pred = (
                val_pred_flat[..., 0].contiguous().view(num_val_samples, Nx, Ny, Nz)
            )
            val_loss_f = forward_criterion(val_pred, val_target_all).item()

        loss_history["val_loss"].append(val_loss_f)
        scheduler_f.step(val_loss_f)

        new_lr = forward_optimizer.param_groups[0]["lr"]
        epoch_time = time.perf_counter() - epoch_start

        print(
            f"Epoch {epoch + 1:03d} done | "
            f"train_loss={train_loss_f:.6e} | "
            f"val_loss={val_loss_f:.6e} | "
            f"best_val={best_val_loss_f:.6e} | "
            f"epoch_time={epoch_time:.2f} s"
        )

        if new_lr != current_lr:
            print(f"Learning rate reduced: {current_lr:.2e} -> {new_lr:.2e}")

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
                output_dir / "best_test_forward_model_PlainMLP.pth",
            )
            print("✓ New best model saved.")
        else:
            epochs_without_improvement += 1
            print(
                f"No improvement: {epochs_without_improvement}/{patience} "
                f"epochs without progress"
            )

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    print("Training complete.")

    with Path(output_dir / "loss_history.json").open("w", encoding="utf-8") as f:
        json.dump(loss_history, f, indent=4)
    print(f"--> Saved metrics data to: {output_dir / 'loss_history_PlainMLP.json'}")

    plot_training_convergence(
        loss_history, output_dir / "convergence_curve_test_PlainMLP.png"
    )

    torch.save(forward_model.state_dict(), "data/15/test_model_PlainMLP.pth")


def test_PureSIREN() -> None:
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
    params_norm = np.array([0.3, 0.6, 0.8, 0.6, 0.4, 0.5], dtype=np.float64)

    # If you already have physical parameters, skip denormalize_params
    # and adapt simulate_function_on_grid accordingly.

    # ------------------------------------------------------------------
    # 2. Build a fresh test grid
    # ------------------------------------------------------------------
    Nx, Ny, Nz = 160, 160, 200
    x_grid = np.linspace(-1.0, 1.0, Nx, dtype=np.float32)
    y_grid = np.linspace(-1.0, 1.0, Ny, dtype=np.float32)
    z_grid = np.linspace(-1.0, 1.0, Nz, dtype=np.float32)

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
    forward_model = PureSIREN(
        param_dim=6,
        output_dim=1,
    ).to(DEVICE)

    checkpoint = torch.load(
        "data/15/best_test_forward_model_Pure_SIREN.pth",
        map_location=DEVICE,
    )

    forward_model.load_state_dict(checkpoint["model_state_dict"])
    forward_model.eval()

    # ------------------------------------------------------------------
    # 4. Ground truth and prediction
    # ------------------------------------------------------------------
    actual_field = simulate_function(
        params_norm,
        torch.from_numpy(x_grid),
        torch.from_numpy(y_grid),
        torch.from_numpy(z_grid),
    )

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
    center_x = Nx // 2 + 30
    center_y = Ny // 2 + 30
    center_z = Nz // 2 + 50

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
        "data/15/simple_field_prediction_test_PureSIREN.png",
        bbox_inches="tight",
        dpi=300,
    )
    # ============================================================
    # Global quantitative metrics
    # ============================================================

    diff = pred_field - actual_field

    mse = np.mean(diff**2)
    rmse = np.sqrt(mse)

    mae = np.mean(np.abs(diff))

    target_rms = np.sqrt(np.mean(actual_field**2))
    relative_rmse = rmse / (target_rms + 1e-12)

    l2_error = np.linalg.norm(diff.ravel())
    l2_target = np.linalg.norm(actual_field.ravel())
    relative_l2 = l2_error / (l2_target + 1e-12)

    max_error = np.max(np.abs(diff))

    # Correlation coefficient
    corr = np.corrcoef(
        actual_field.ravel(),
        pred_field.ravel(),
    )[0, 1]

    # R^2 coefficient
    ss_res = np.sum(diff**2)
    ss_tot = np.sum((actual_field - np.mean(actual_field)) ** 2)
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)

    print("\n================ GLOBAL METRICS ================")
    print(f"MSE           : {mse:.6e}")
    print(f"RMSE          : {rmse:.6e}")
    print(f"MAE           : {mae:.6e}")
    print(f"Relative RMSE : {relative_rmse:.6e}")
    print(f"Relative L2   : {relative_l2:.6e}")
    print(f"Max Error     : {max_error:.6e}")
    print(f"Correlation   : {corr:.6f}")
    print(f"R^2           : {r2:.6f}")
    print("================================================\n")

    print("--> Simple field comparison plot saved successfully.")


def test_CondSIREN() -> None:
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
    params_norm = np.array([0.3, 0.6, 0.8, 0.6, 0.4, 0.5], dtype=np.float64)

    # If you already have physical parameters, skip denormalize_params
    # and adapt simulate_function_on_grid accordingly.

    # ------------------------------------------------------------------
    # 2. Build a fresh test grid
    # ------------------------------------------------------------------
    Nx, Ny, Nz = 160, 160, 200
    x_grid = np.linspace(-1.0, 1.0, Nx, dtype=np.float32)
    y_grid = np.linspace(-1.0, 1.0, Ny, dtype=np.float32)
    z_grid = np.linspace(-1.0, 1.0, Nz, dtype=np.float32)

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
        param_dim=6,
        output_dim=1,
    ).to(DEVICE)

    checkpoint = torch.load(
        "data/15/best_test_forward_model_CondSIREN.pth",
        map_location=DEVICE,
    )

    forward_model.load_state_dict(checkpoint["model_state_dict"])
    forward_model.eval()

    # ------------------------------------------------------------------
    # 4. Ground truth and prediction
    # ------------------------------------------------------------------
    actual_field = simulate_function(
        params_norm,
        torch.from_numpy(x_grid),
        torch.from_numpy(y_grid),
        torch.from_numpy(z_grid),
    )

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
    center_x = Nx // 2 + 50
    center_y = Ny // 2 + 50
    center_z = Nz // 2 + 5

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
        "data/15/simple_field_prediction_test_CondSIREN.png",
        bbox_inches="tight",
        dpi=300,
    )
    # ============================================================
    # Global quantitative metrics
    # ============================================================

    diff = pred_field - actual_field

    mse = np.mean(diff**2)
    rmse = np.sqrt(mse)

    mae = np.mean(np.abs(diff))

    target_rms = np.sqrt(np.mean(actual_field**2))
    relative_rmse = rmse / (target_rms + 1e-12)

    l2_error = np.linalg.norm(diff.ravel())
    l2_target = np.linalg.norm(actual_field.ravel())
    relative_l2 = l2_error / (l2_target + 1e-12)

    max_error = np.max(np.abs(diff))

    # Correlation coefficient
    corr = np.corrcoef(
        actual_field.ravel(),
        pred_field.ravel(),
    )[0, 1]

    # R^2 coefficient
    ss_res = np.sum(diff**2)
    ss_tot = np.sum((actual_field - np.mean(actual_field)) ** 2)
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)

    print("\n================ GLOBAL METRICS ================")
    print(f"MSE           : {mse:.6e}")
    print(f"RMSE          : {rmse:.6e}")
    print(f"MAE           : {mae:.6e}")
    print(f"Relative RMSE : {relative_rmse:.6e}")
    print(f"Relative L2   : {relative_l2:.6e}")
    print(f"Max Error     : {max_error:.6e}")
    print(f"Correlation   : {corr:.6f}")
    print(f"R^2           : {r2:.6f}")
    print("================================================\n")

    print("--> Simple field comparison plot saved successfully.")


def test_PlainMLP() -> None:
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
    params_norm = np.array([0.3, 0.6, 0.8, 0.6, 0.4, 0.5], dtype=np.float64)

    # If you already have physical parameters, skip denormalize_params
    # and adapt simulate_function_on_grid accordingly.

    # ------------------------------------------------------------------
    # 2. Build a fresh test grid
    # ------------------------------------------------------------------
    Nx, Ny, Nz = 160, 160, 200
    x_grid = np.linspace(-1.0, 1.0, Nx, dtype=np.float32)
    y_grid = np.linspace(-1.0, 1.0, Ny, dtype=np.float32)
    z_grid = np.linspace(-1.0, 1.0, Nz, dtype=np.float32)

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
    forward_model = PureMLP(
        param_dim=6,
        output_dim=1,
    ).to(DEVICE)

    checkpoint = torch.load(
        "data/15/best_test_forward_model_PlainMLP.pth",
        map_location=DEVICE,
    )

    forward_model.load_state_dict(checkpoint["model_state_dict"])
    forward_model.eval()

    # ------------------------------------------------------------------
    # 4. Ground truth and prediction
    # ------------------------------------------------------------------
    actual_field = simulate_function(
        params_norm,
        torch.from_numpy(x_grid),
        torch.from_numpy(y_grid),
        torch.from_numpy(z_grid),
    )

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
    center_x = Nx // 2 + 50
    center_y = Ny // 2 + 50
    center_z = Nz // 2 + 5

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
        "data/15/simple_field_prediction_test_PlainMLP.png",
        bbox_inches="tight",
        dpi=300,
    )
    # ============================================================
    # Global quantitative metrics
    # ============================================================

    diff = pred_field - actual_field

    mse = np.mean(diff**2)
    rmse = np.sqrt(mse)

    mae = np.mean(np.abs(diff))

    target_rms = np.sqrt(np.mean(actual_field**2))
    relative_rmse = rmse / (target_rms + 1e-12)

    l2_error = np.linalg.norm(diff.ravel())
    l2_target = np.linalg.norm(actual_field.ravel())
    relative_l2 = l2_error / (l2_target + 1e-12)

    max_error = np.max(np.abs(diff))

    # Correlation coefficient
    corr = np.corrcoef(
        actual_field.ravel(),
        pred_field.ravel(),
    )[0, 1]

    # R^2 coefficient
    ss_res = np.sum(diff**2)
    ss_tot = np.sum((actual_field - np.mean(actual_field)) ** 2)
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)

    print("\n================ GLOBAL METRICS ================")
    print(f"MSE           : {mse:.6e}")
    print(f"RMSE          : {rmse:.6e}")
    print(f"MAE           : {mae:.6e}")
    print(f"Relative RMSE : {relative_rmse:.6e}")
    print(f"Relative L2   : {relative_l2:.6e}")
    print(f"Max Error     : {max_error:.6e}")
    print(f"Correlation   : {corr:.6f}")
    print(f"R^2           : {r2:.6f}")
    print("================================================\n")

    print("--> Simple field comparison plot saved successfully.")


def _load_model_checkpoint(model: torch.nn.Module, ckpt_path: str | Path) -> None:
    ckpt = torch.load(ckpt_path, map_location=DEVICE)

    # Support both {"model_state_dict": ...} and raw state_dict checkpoints.
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict)
    model.eval()


def _make_pred_field(
    model: torch.nn.Module,
    params_norm: np.ndarray,
    coords: torch.Tensor,
    Nx: int,
    Ny: int,
    Nz: int,
) -> np.ndarray:
    with torch.no_grad():
        param_tensor = torch.tensor(
            params_norm,
            dtype=torch.float32,
            device=DEVICE,
        ).unsqueeze(0)  # (1, 3)

        pred_field_batch = model(param_tensor, coords)  # (1, N_pts, 1)

        if pred_field_batch.ndim != 3 or pred_field_batch.shape[-1] != 1:
            msg = f"Unexpected model output shape: {pred_field_batch.shape}"
            raise ValueError(msg)

        return pred_field_batch[0, :, 0].reshape(Nx, Ny, Nz).cpu().numpy()


def _load_loss_history(history_path: str | Path) -> dict:
    with Path(history_path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _plot_val_loss_curves(
    histories: dict[str, dict],
    save_path: str | Path,
) -> None:
    plt.style.use(
        "seaborn-v0_8-whitegrid"
        if "seaborn-v0_8-whitegrid" in plt.style.available
        else "default"
    )

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)

    line_styles = {
        "CondSIREN": {"color": "#1f77b4", "linestyle": "-", "linewidth": 2},
        "PureSIREN": {"color": "#ff7f0e", "linestyle": "--", "linewidth": 2},
        "PlainMLP": {"color": "#2ca02c", "linestyle": "-.", "linewidth": 2},
    }

    for name, history in histories.items():
        val_loss = history["val_loss"]
        epochs = range(1, len(val_loss) + 1)
        ax.plot(epochs, val_loss, label=name, **line_styles.get(name, {}))

    ax.set_yscale("log")
    ax.set_xlabel("Epochs", fontsize=12, fontweight="bold")
    ax.set_ylabel("Validation Loss (Log Scale)", fontsize=12, fontweight="bold")
    ax.set_title("Validation Loss Curves", fontsize=14, fontweight="bold")
    ax.legend(frameon=True, facecolor="white")
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"--> Saved validation loss plot to: {save_path}")


def test_all(
    params_norm: np.ndarray = np.array(
        [0.3, 0.6, 0.8, 0.6, 0.4, 0.5], dtype=np.float64
    ),
    grid_shape: tuple[int, int, int] = (160, 160, 200),
    output_dir: str | Path = Path("data/15"),
    cond_ckpt: str | Path = "data/15/best_test_forward_model_CondSIREN.pth",
    pure_ckpt: str | Path = "data/15/best_test_forward_model_PureSIREN.pth",
    mlp_ckpt: str | Path = "data/15/best_test_forward_model_PlainMLP.pth",
    cond_history_path: str | Path = "data/15/loss_history_CondSIREN.json",
    pure_history_path: str | Path = "data/15/loss_history_PureSIREN.json",
    mlp_history_path: str | Path = "data/15/loss_history_PlainMLP.json",
) -> None:
    """
    Compare Actual vs CondSIREN vs PureSIREN vs PlainMLP.

    Produces:
      1) 1D z-slice plot with all three predictions
      2) 2D slice figure with 4 panels: actual, PureSIREN, CondSIREN, PlainMLP
      3) Validation-loss comparison plot for the 3 models
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    Nx, Ny, Nz = grid_shape
    x_grid = np.linspace(-1.0, 1.0, Nx, dtype=np.float32)
    y_grid = np.linspace(-1.0, 1.0, Ny, dtype=np.float32)
    z_grid = np.linspace(-1.0, 1.0, Nz, dtype=np.float32)

    X, Y, Z = torch.meshgrid(
        torch.from_numpy(x_grid),
        torch.from_numpy(y_grid),
        torch.from_numpy(z_grid),
        indexing="ij",
    )

    coords = (
        torch.stack(
            (X.reshape(-1), Y.reshape(-1), Z.reshape(-1)),
            dim=-1,
        )
        .to(DEVICE)
        .float()
    )

    # ----------------------------
    # Ground truth
    # ----------------------------
    actual_field = simulate_function(
        params_norm,
        torch.from_numpy(x_grid),
        torch.from_numpy(y_grid),
        torch.from_numpy(z_grid),
    )

    # ----------------------------
    # Load models
    # ----------------------------
    cond_model = ForwardCondSIRENStateModel(
        param_dim=6,
        output_dim=1,
    ).to(DEVICE)

    pure_model = PureSIREN(
        param_dim=6,
        coord_dim=3,
        output_dim=1,
    ).to(DEVICE)

    mlp_model = PureMLP(
        param_dim=6,
        coord_dim=3,
        output_dim=1,
    ).to(DEVICE)

    _load_model_checkpoint(cond_model, cond_ckpt)
    _load_model_checkpoint(pure_model, pure_ckpt)
    _load_model_checkpoint(mlp_model, mlp_ckpt)

    # ----------------------------
    # Predictions
    # ----------------------------
    cond_field = _make_pred_field(cond_model, params_norm, coords, Nx, Ny, Nz)
    pure_field = _make_pred_field(pure_model, params_norm, coords, Nx, Ny, Nz)
    mlp_field = _make_pred_field(mlp_model, params_norm, coords, Nx, Ny, Nz)

    # ----------------------------
    # Slice locations
    # ----------------------------
    center_x = Nx // 2
    center_y = Ny // 2
    center_z = Nz // 2

    actual_z = actual_field[center_x, center_y, :]
    cond_z = cond_field[center_x, center_y, :]
    pure_z = pure_field[center_x, center_y, :]
    mlp_z = mlp_field[center_x, center_y, :]

    actual_xy = actual_field[:, :, center_z]
    cond_xy = cond_field[:, :, center_z]
    pure_xy = pure_field[:, :, center_z]
    mlp_xy = mlp_field[:, :, center_z]

    extent = [x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]]

    # ============================================================
    # 1) 1D slice plot
    # ============================================================
    plt.style.use(
        "seaborn-v0_8-whitegrid"
        if "seaborn-v0_8-whitegrid" in plt.style.available
        else "default"
    )

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    ax.plot(z_grid, actual_z, label="Actual", linewidth=2)
    ax.plot(z_grid, cond_z, "--", label="CondSIREN", linewidth=2)
    ax.plot(z_grid, pure_z, "-.", label="PureSIREN", linewidth=2)
    ax.plot(z_grid, mlp_z, ":", label="PlainMLP", linewidth=2.5)

    ax.set_xlabel("z")
    ax.set_ylabel("f(x0, y0, z)")
    ax.set_title(f"1D slice at x={x_grid[center_x]:.3f}, y={y_grid[center_y]:.3f}")
    ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(
        output_dir / "compare_1d_slice_all_models.png", bbox_inches="tight", dpi=300
    )
    plt.close(fig)
    print(
        f"--> Saved 1D comparison plot to: {output_dir / 'compare_1d_slice_all_models.png'}"
    )

    # ============================================================
    # 2) 2D slice plot (4 panels)
    # ============================================================
    vmin = min(
        actual_xy.min(),
        cond_xy.min(),
        pure_xy.min(),
        mlp_xy.min(),
    )
    vmax = max(
        actual_xy.max(),
        cond_xy.max(),
        pure_xy.max(),
        mlp_xy.max(),
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=300)

    panels = [
        ("Actual", actual_xy, axes[0, 0]),
        ("PureSIREN", pure_xy, axes[0, 1]),
        ("CondSIREN", cond_xy, axes[1, 0]),
        ("PlainMLP", mlp_xy, axes[1, 1]),
    ]

    for title, data, ax in panels:
        im = ax.imshow(
            data.T,
            origin="lower",
            aspect="auto",
            extent=extent,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"{title} x-y slice at z={z_grid[center_z]:.3f}")
        fig.colorbar(im, ax=ax, shrink=0.85)

    plt.tight_layout()
    fig.savefig(
        output_dir / "compare_2d_slices_all_models.png", bbox_inches="tight", dpi=300
    )
    plt.close(fig)
    print(
        f"--> Saved 2D comparison plot to: {output_dir / 'compare_2d_slices_all_models.png'}"
    )

    # ============================================================
    # 3) Validation loss curves
    # ============================================================
    histories = {
        "CondSIREN": _load_loss_history(cond_history_path),
        "PureSIREN": _load_loss_history(pure_history_path),
        "PlainMLP": _load_loss_history(mlp_history_path),
    }
    _plot_val_loss_curves(
        histories,
        output_dir / "validation_loss_all_models.png",
    )

    # Optional: print global metrics for each model
    def _metrics(pred: np.ndarray, name: str) -> None:
        diff = pred - actual_field
        mse = np.mean(diff**2)
        rmse = np.sqrt(mse)
        rel_l2 = np.linalg.norm(diff.ravel()) / (
            np.linalg.norm(actual_field.ravel()) + 1e-12
        )
        corr = np.corrcoef(actual_field.ravel(), pred.ravel())[0, 1]
        r2 = 1.0 - np.sum(diff**2) / (
            np.sum((actual_field - np.mean(actual_field)) ** 2) + 1e-12
        )
        print(f"\n[{name}]")
        print(f"MSE   : {mse:.6e}")
        print(f"RMSE  : {rmse:.6e}")
        print(f"RelL2 : {rel_l2:.6e}")
        print(f"Corr  : {corr:.6f}")
        print(f"R^2   : {r2:.6f}")

    _metrics(cond_field, "CondSIREN")
    _metrics(pure_field, "PureSIREN")
    _metrics(mlp_field, "PlainMLP")

    print("\n--> All-model comparison complete.")


if __name__ == "__main__":
    RUN_TRAIN_PureSIREN = False
    RUN_TRAIN_CondSIREN = True
    RUN_TRAIN_PlainMLP = False
    RUN_TEST_PureSIREN = False
    RUN_TEST_CondSIREN = True
    RUN_TEST_PlainMLP = False
    RUN_TEST_ALL = False

    if RUN_TRAIN_PureSIREN:
        train_PureSIREN()
    if RUN_TEST_PureSIREN:
        test_PureSIREN()
    if RUN_TRAIN_CondSIREN:
        train_CondSIREN()
    if RUN_TEST_CondSIREN:
        test_CondSIREN()
    if RUN_TRAIN_PlainMLP:
        train_PlainML()
    if RUN_TEST_PlainMLP:
        test_PlainMLP()
    if RUN_TEST_ALL:
        test_all()
