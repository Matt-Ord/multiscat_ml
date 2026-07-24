import contextlib
import json
import time
from pathlib import Path
from typing import Any, override

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")  # pyright: ignore[reportConstantRedefinition]
else:
    DEVICE = torch.device("cpu")  # pyright: ignore[reportConstantRedefinition]

_BOUNDS = {
    "x": (-4.0, 4.0),
    "y": (-4.0, 4.0),
    "z": (-4.0, 6.0),
    "kx": (-4.0, 4.0),
    "ky": (-4.0, 4.0),
    "kz": (-4.0, 4.0),
}


def _test_function(params: torch.Tensor) -> torch.Tensor:
    x, y, z, asymptote, ky, kz = params

    # Fundamental spatial frequency for domain [-4, 4] (period = 8)

    z_start = -2.0 + 0.2 * kz  # Region where function departs from 0 (around z = -2)
    z_flat = 3.25 + 0.25 * kz  # Plateau region where it flattens out (z = 3.5 to 5)
    width = z_flat - z_start
    z_mid = 0.5 * (z_start + z_flat)

    # 2. Transition Envelopes
    gamma = 1.0 + 0.2 * torch.nn.functional.softplus(ky)
    sigma = torch.sigmoid((6.0 / width) * gamma * (z - z_mid))
    envelope = torch.exp(-6.0 * ((z - z_mid) / width) ** 2)

    # 3. Fourier Spatial Channels
    w0 = torch.pi / 4.0
    c_persistent = asymptote * (
        0.5 * torch.cos(w0 * x) * torch.cos(w0 * y)
        + 0.5 * torch.cos(w0 * x + 0.21) * torch.cos(2.0 * w0 * y - 0.31)
    )
    c_decaying = torch.sin(2.0 * w0 * x) + torch.cos(2.0 * w0 * y)

    # 4. Multiplicative z-Oscillation Factor
    raw_z_oscillation = 2.5 * torch.cos(
        8.0 * torch.pi * (z - z_start) / width
    ) + 1.5 * torch.sin(3.0 * torch.pi * (z - z_start) / width) * torch.sin(
        w0 * x
    ) * torch.cos(w0 * y)

    # Combine channels with multiplicative oscillation
    open_channel = (sigma) * c_persistent * (1.0 - envelope * raw_z_oscillation)
    transient_channel = envelope * 0.5 * c_decaying

    return torch.sigmoid(4.0 * (z - z_start)) * (open_channel + transient_channel)


def _generate_parameters(n_samples: int) -> torch.Tensor:
    """Generate a 6 * n_sample array of random parameters within the specified bounds."""
    lows = torch.tensor([b[0] for b in _BOUNDS.values()], device=DEVICE).unsqueeze(1)
    highs = torch.tensor([b[1] for b in _BOUNDS.values()], device=DEVICE).unsqueeze(1)

    # Uniformly sample in [0, 1) and scale to [low, high)
    unscaled = torch.rand(len(_BOUNDS), n_samples, device=DEVICE)
    return lows + (highs - lows) * unscaled


def _generate_dataset(n_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a dataset of random parameters and their corresponding targets."""
    params = _generate_parameters(n_samples)
    targets = _test_function(params)
    return params.T, targets  # Returns (n_samples, 6) and (n_samples,)


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


class SirenLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        is_first: bool = False,
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
    def __init__(
        self,
        param_dim: int = 3,
        coord_dim: int = 3,
        cond_dim: int = 64,
        hidden_dim: int = 128,
        output_dim: int = 1,
        num_siren_layers: int = 4,
        first_omega_0: float = 5.0,
        hidden_omega_0: float = 5.0,
    ) -> None:
        super().__init__()

        if param_dim == 6 and coord_dim == 3:
            param_dim = 3

        self.param_dim = param_dim
        self.coord_dim = coord_dim
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

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 6) -> 3 coordinates + 3 physical parameters
        returns: (B, output_dim).
        """
        coords = x[:, : self.coord_dim]
        params = x[:, self.coord_dim :]

        cond = self.condition_encoder(params)

        for layer in self.siren_layers:
            coords = layer(coords, cond)

        return self.head(coords)


class PureSIREN(nn.Module):
    """
    Input:
        x: (B, 6).

    Output:
        (B, output_dim)
    """

    def __init__(
        self,
        in_dim: int = 6,
        param_dim: int | None = None,
        coord_dim: int = 0,
        hidden_dim: int = 256,
        num_layers: int = 6,
        first_omega_0: float = 5.0,
        hidden_omega_0: float = 5.0,
        output_dim: int = 1,
    ) -> None:
        super().__init__()

        if param_dim is not None:
            in_dim = param_dim + coord_dim if coord_dim > 0 else param_dim

        self.in_dim = in_dim
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

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        for layer in self.net:
            x = layer(x)

        return self.head(x)


class ResBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout_rate: float = 0.05) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout_rate),
        )
        self.act = nn.GELU()

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class PureMLP(nn.Module):
    """
    Plain MLP baseline with the same interface as PureSIREN.

    Input:
        x : (B, 6)

    Output:
        (B, output_dim)
    """

    def __init__(
        self,
        in_dim: int = 6,
        param_dim: int | None = None,
        coord_dim: int = 0,
        hidden_dim: int = 128,
        num_blocks: int = 5,
        output_dim: int = 1,
    ) -> None:
        super().__init__()

        if param_dim is not None:
            in_dim = param_dim + coord_dim if coord_dim > 0 else param_dim

        self.output_dim = output_dim

        self.embedding = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
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

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.res_blocks(x)
        return self.head(x)


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
    plt.style.use(
        "seaborn-v0_8-whitegrid"
        if "seaborn-v0_8-whitegrid" in plt.style.available
        else "default"
    )

    _fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    epochs_range = range(1, len(history["train_loss"]) + 1)

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

    ax.set_yscale("log")
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


def train_model(
    model: nn.Module,
    epochs: tuple[int, int, int] = (200, 100, 100),
    max_epochs_without_improvement: int = 200,
    output_dir: Path = Path("data/15"),
) -> None:

    output_dir.mkdir(parents=True, exist_ok=True)

    model = model.to(DEVICE)
    forward_criterion = nn.MSELoss()
    forward_optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        forward_optimizer, mode="min", factor=0.5, patience=5
    )

    loss_history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    n_epochs, n_batch, _n_per_batch = epochs

    full_dataset = TensorDataset(*_generate_dataset(n_samples=1_000_000))
    train_dataset, val_dataset = random_split(full_dataset, [0.8, 0.2])

    train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2048, shuffle=False)

    for epoch in range(n_epochs):
        epoch_start = time.perf_counter()
        model.train()
        train_loss = 0.0
        current_lr = forward_optimizer.param_groups[0]["lr"]

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{n_epochs}",
            unit="batch",
        )

        for batch_idx, (batch_parameters, batch_targets) in enumerate(pbar):
            forward_optimizer.zero_grad(set_to_none=True)

            prediction = model(batch_parameters).squeeze(-1)
            batch_loss = forward_criterion(prediction, batch_targets)
            batch_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            forward_optimizer.step()

            train_loss += batch_loss.item()
            running_loss = train_loss / (batch_idx + 1)

            pbar.set_postfix(
                loss=f"{batch_loss.item():.3e}",
                avg=f"{running_loss:.3e}",
                lr=f"{current_lr:.1e}",
            )

        train_loss /= n_batch
        loss_history["train_loss"].append(train_loss)

        # Validation step
        model.eval()
        validation_loss = 0.0
        with torch.no_grad():
            for val_params, val_targets in val_loader:
                prediction = model(val_params).squeeze(-1)
                validation_loss += forward_criterion(prediction, val_targets).item()

        loss_history["val_loss"].append(validation_loss)
        scheduler.step(validation_loss)

        epoch_time = time.perf_counter() - epoch_start
        print(
            f"Epoch {epoch + 1:03d} done | train_loss={train_loss:.6e} | "
            f"val_loss={validation_loss:.6e} | best_val={best_val_loss:.6e} | time={epoch_time:.2f}s"
        )

        if validation_loss < best_val_loss:
            best_val_loss = validation_loss
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": forward_optimizer.state_dict(),
                    "epoch": epoch,
                    "val_loss": validation_loss,
                },
                output_dir / "best_model.pth",
            )
            print("✓ Saved new best model checkpoint.")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= max_epochs_without_improvement:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    # Save artifacts
    loss_history_path = output_dir / "loss_history.json"
    with loss_history_path.open("w", encoding="utf-8") as f:
        json.dump(loss_history, f, indent=4)

    plot_training_convergence(loss_history, output_dir / "convergence_curve.png")
    torch.save(model.state_dict(), output_dir / "final_model.pth")


def get_target_against_z(
    coordinates: tuple[float, float] = (1.0, 1.0),
    parameters: tuple[float, float, float] = (2.0, 2.0, 2.0),
    z_points: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the ground truth target function f(z) over a z-range for fixed (x, y) and wavevectors."""
    if z_points is None:
        z_points = torch.linspace(_BOUNDS["z"][0], _BOUNDS["z"][1], 100, device=DEVICE)

    n_pts = len(z_points)
    x_t = torch.full((n_pts,), coordinates[0], device=DEVICE)
    y_t = torch.full((n_pts,), coordinates[1], device=DEVICE)
    kx_t = torch.full((n_pts,), parameters[0], device=DEVICE)
    ky_t = torch.full((n_pts,), parameters[1], device=DEVICE)
    kz_t = torch.full((n_pts,), parameters[2], device=DEVICE)

    # Shape expected by _test_function: (6, N)
    params = torch.stack([x_t, y_t, z_points, kx_t, ky_t, kz_t], dim=0)
    targets = _test_function(params)

    return z_points, targets


def get_prediction_against_z(
    model: nn.Module,
    coordinates: tuple[float, float] = (1.0, 1.0),
    parameters: tuple[float, float, float] = (2.0, 2.0, 2.0),
    z_points: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate predictions f(z) from a model over a z-range for fixed (x, y) and wavevectors."""
    if z_points is None:
        z_points = torch.linspace(_BOUNDS["z"][0], _BOUNDS["z"][1], 100, device=DEVICE)

    n_pts = len(z_points)
    x_t = torch.full((n_pts,), coordinates[0], device=DEVICE)
    y_t = torch.full((n_pts,), coordinates[1], device=DEVICE)
    kx_t = torch.full((n_pts,), parameters[0], device=DEVICE)
    ky_t = torch.full((n_pts,), parameters[1], device=DEVICE)
    kz_t = torch.full((n_pts,), parameters[2], device=DEVICE)

    # Model input shape: (N, 6) -> [x, y, z, kx, ky, kz]
    inputs = torch.stack([x_t, y_t, z_points, kx_t, ky_t, kz_t], dim=1)

    model.eval()
    with torch.no_grad():
        predictions = model(inputs).squeeze(-1)

    return z_points, predictions


def compare_models_against_z(
    model_zoo: dict[str, nn.Module],
    coordinates: tuple[float, float] = (1.0, 1.0),
    parameters: tuple[float, float, float] = (2.0, 2.0, 2.0),
) -> None:
    """Plot and compares ground truth target vs predictions from multiple models along the z-axis."""
    delta_z = _BOUNDS["z"][1] - _BOUNDS["z"][0]
    z_points = torch.linspace(
        _BOUNDS["z"][0] - 0.5 * delta_z,
        _BOUNDS["z"][1] + 0.5 * delta_z,
        500,
        device=DEVICE,
    )

    # 1. Compute ground truth
    _, targets = get_target_against_z(
        coordinates=coordinates, parameters=parameters, z_points=z_points
    )

    plt.style.use(
        "seaborn-v0_8-whitegrid"
        if "seaborn-v0_8-whitegrid" in plt.style.available
        else "default"
    )
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

    z_np = z_points.cpu().numpy()
    ax.plot(
        z_np,
        targets.cpu().numpy(),
        label="Target (Ground Truth)",
        color="black",
        linewidth=2.0,
    )

    # 2. Compute predictions for each model in model_zoo
    for name, model in model_zoo.items():
        _, preds = get_prediction_against_z(
            model=model,
            coordinates=coordinates,
            parameters=parameters,
            z_points=z_points,
        )
        ax.plot(
            z_np,
            preds.cpu().numpy(),
            label=f"Pred: {name}",
            linestyle="--",
            linewidth=1.5,
        )

    ax.set_xlabel("z", fontsize=12, fontweight="bold")
    ax.set_ylabel("f(x, y, z, kx, ky, kz)", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Model Comparison vs z-axis\n(x={coordinates[0]}, y={coordinates[1]}, kx={parameters[0]}, ky={parameters[1]}, kz={parameters[2]})",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(frameon=True, facecolor="white", edgecolor="none")
    ax.set_xlim(z_np[0], z_np[-1])

    ax.axvline(x=_BOUNDS["z"][0], color="gray", linewidth=2.0)
    ax.axvline(x=_BOUNDS["z"][1], color="gray", linewidth=2.0)
    fig.savefig(
        "data/15/model_comparison_vs_z.png",
        bbox_inches="tight",
        dpi=300,
    )


def _load_best_models(
    model_zoo: dict[str, nn.Module], base_path: Path = Path("data/15")
) -> None:
    """Load the best model checkpoints from disk into the provided model zoo."""
    for name, model in model_zoo.items():
        ckpt_path = base_path / name / "best_model.pth"
        if ckpt_path.exists():
            checkpoint = torch.load(ckpt_path, map_location=DEVICE)
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"✓ Loaded trained weights for {name} from {ckpt_path}")


if __name__ == "__main__":
    RUN_TRAIN = False
    RUN_TEST = True
    LOAD_CHECKPOINTS = True

    model_zoo: dict[str, nn.Module] = {
        "PureSIREN": PureSIREN(
            param_dim=6, output_dim=1, first_omega_0=1.0, hidden_omega_0=1.0
        ),
        "CondSIREN": ForwardCondSIRENStateModel(
            param_dim=6, output_dim=1, first_omega_0=1.0, hidden_omega_0=1.0
        ),
        "PlainMLP": PureMLP(param_dim=6, output_dim=1),
    }
    if LOAD_CHECKPOINTS:
        _load_best_models(model_zoo=model_zoo, base_path=Path("data/15"))

    # 1. Train Models Generic Loop
    if RUN_TRAIN:
        for name, model in model_zoo.items():
            print(f"\n[{name}] Training started on device: {DEVICE}")
            train_model(
                model=model,
                epochs=(200, 100, 100),
                output_dir=Path(f"data/15/{name}"),
            )
    if RUN_TEST:
        compare_models_against_z(
            model_zoo=model_zoo,
            coordinates=(1, 1),
            parameters=(2.0, 2.0, 2.0),
        )
