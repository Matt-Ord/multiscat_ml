import time
from pathlib import Path
from typing import TYPE_CHECKING, override

import numpy as np
import torch
from slate_core.plot import get_figure
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

from multiscat_ml import plot_loss_curves
from multiscat_ml.model_zoo import ModelZooEntry, compare_model_validation_loss
from multiscat_ml.utils import (
    TrainingStats,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")  # pyright: ignore[reportConstantRedefinition]
else:
    DEVICE = torch.device("cpu")  # pyright: ignore[reportConstantRedefinition]

_BOUNDS = {
    "x": (-4.0, 4.0),
    "y": (-4.0, 4.0),
    "z": (-4.0, 8.0),
    "kx": (-4.0, 4.0),
    "ky": (-4.0, 4.0),
    "kz": (-4.0, 4.0),
}


def _test_function(params: torch.Tensor) -> torch.Tensor:  # ruff: ignore[too-many-locals]
    x, y, z, asymptote, ky, kz = params

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
        is_first: bool = False,  # ruff: ignore[boolean-default-value-positional-argument, boolean-type-hint-positional-argument]
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

            # Start close to an un-modulated SIREN
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
    def __init__(  # ruff: ignore[too-many-arguments, too-many-positional-arguments]
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

        if param_dim == 6 and coord_dim == 3:  # ruff: ignore[magic-value-comparison]
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
        coords = x[:, : self.coord_dim]
        params = x[:, self.coord_dim :]

        cond = self.condition_encoder(params)

        for layer in self.siren_layers:
            coords = layer(coords, cond)

        return self.head(coords)


class PureSIREN(nn.Module):
    def __init__(  # ruff: ignore[too-many-arguments, too-many-positional-arguments]
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

    def __init__(  # ruff: ignore[too-many-arguments, too-many-positional-arguments]
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


# TODO: we haven't considered the fact that the  # ruff: ignore[line-contains-todo]
# region of oscillation depends on both (kx, ky, kz)
# and on the channel idx
class ExplicitAsymptoticGaborNet(nn.Module):
    def __init__(  # ruff: ignore[too-many-arguments, too-many-positional-arguments]
        self,
        in_dim: int = 6,
        param_dim: int = 3,
        hidden_dim: int = 128,
        omega_0: float = 3.0,
        sigma_0: float = 1.5,
        output_dim: int = 1,
    ) -> None:
        super().__init__()

        # 1. Persistent 2D Stream
        self.persistent_net = nn.Sequential(
            SirenLayer(
                in_features=2 + param_dim,
                out_features=hidden_dim,
                is_first=True,
                omega_0=omega_0,
            ),
            SirenLayer(
                in_features=hidden_dim,
                out_features=hidden_dim,
                is_first=False,
                omega_0=omega_0,
            ),
            nn.Linear(hidden_dim, output_dim),
        )

        # 2. Parameter-Agnostic z-Gate
        self.z_gate = nn.Sequential(
            nn.Linear(1 + param_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

        # 3. Localized Transient Gabor Stream
        self.freq_linear = nn.Linear(in_dim, hidden_dim)
        self.envelope_linear = nn.Linear(1 + param_dim, hidden_dim)
        self.transient_head = nn.Linear(hidden_dim, output_dim)

        self.omega_0 = omega_0
        self.sigma_0 = sigma_0
        self.init_weights()

    def init_weights(self) -> None:
        with torch.no_grad():
            bound = np.sqrt(6.0 / self.freq_linear.in_features) / self.omega_0
            self.freq_linear.weight.uniform_(-bound, bound)
            self.freq_linear.bias.uniform_(-bound, bound)

            nn.init.normal_(self.envelope_linear.weight, std=0.1)
            nn.init.zeros_(self.envelope_linear.bias)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Deconstruct inputs
        z = x[:, 2:3]
        params = x[:, 3:]  # [kx, ky, kz]

        spatial_inputs = torch.cat([x[:, :2], params], dim=-1)
        c_persistent = self.persistent_net(spatial_inputs)

        z_and_params = torch.cat([z, params], dim=-1)
        gate = torch.sigmoid(self.z_gate(z_and_params))

        freq = self.omega_0 * self.freq_linear(x)
        envelope = self.sigma_0 * self.envelope_linear(z_and_params)

        gabor_feats = torch.sin(freq) * torch.exp(-0.5 * (envelope**2))
        y_transient = self.transient_head(gabor_feats)

        return gate * c_persistent + y_transient


class ExplicitAsymptoticGaborNet1(nn.Module):
    def __init__(
        self,
        in_dim: int = 6,
        param_dim: int = 3,
        hidden_dim: int = 128,
        omega_0: float = 3.0,
        output_dim: int = 1,
    ) -> None:
        super().__init__()

        # 1. Persistent 2D Stream
        self.persistent_net = nn.Sequential(
            SirenLayer(
                in_features=2 + param_dim,
                out_features=hidden_dim,
                is_first=True,
                omega_0=omega_0,
            ),
            SirenLayer(
                in_features=hidden_dim,
                out_features=hidden_dim,
                is_first=False,
                omega_0=omega_0,
            ),
            nn.Linear(hidden_dim, output_dim),
        )

        # 2. Parameter-Agnostic z-Gate
        self.z_gate = nn.Sequential(
            nn.Linear(1 + param_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

        # 3. Explicit Localized Gabor Stream
        self.freq_linear = nn.Linear(in_dim, hidden_dim)

        self.center_net = nn.Linear(param_dim, hidden_dim)
        self.width_net = nn.Linear(param_dim, hidden_dim)

        self.transient_head = nn.Linear(hidden_dim, output_dim)
        self.omega_0 = omega_0

        self.init_weights()

    def init_weights(self) -> None:
        with torch.no_grad():
            # Standard SIREN bound for frequency linear layer
            bound = np.sqrt(6.0 / self.freq_linear.in_features) / self.omega_0
            self.freq_linear.weight.uniform_(-bound, bound)
            self.freq_linear.bias.uniform_(-bound, bound)

            # Center Gabor envelopes across the actual z domain
            nn.init.uniform_(self.center_net.bias, -2.0, 6.0)
            nn.init.zeros_(self.center_net.weight)

            # Set initial widths to ~1.0 unit in raw z-space
            nn.init.constant_(self.width_net.bias, 0.2)
            nn.init.zeros_(self.width_net.weight)

            head_bound = np.sqrt(6.0 / self.transient_head.in_features) / self.omega_0
            self.transient_head.weight.uniform_(-head_bound, head_bound)
            if self.transient_head.bias is not None:
                self.transient_head.bias.zero_()

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x[:, 2:3]
        params = x[:, 3:]

        # A. Asymptotic Plateau Stream
        spatial_inputs = torch.cat([x[:, :2], params], dim=-1)
        c_persistent = self.persistent_net(spatial_inputs)

        # B. Parameter-Agnostic Gate
        z_and_params = torch.cat([z, params], dim=-1)
        gate = torch.sigmoid(self.z_gate(z_and_params))

        # C. Explicit Center-Width Gabor Stream
        freq = self.omega_0 * self.freq_linear(x)

        centers = self.center_net(params)
        widths = torch.nn.functional.softplus(self.width_net(params)) + 0.2

        envelope = torch.exp(-0.5 * ((z - centers) / widths) ** 2)

        gabor_feats = torch.sin(freq) * envelope
        y_transient = self.transient_head(gabor_feats)

        return gate * c_persistent + y_transient


def train_model(  # ruff: ignore[too-many-locals, too-many-statements]
    model_entry: ModelZooEntry,
    epochs: tuple[int, int, int] = (200, 100, 100),
    max_epochs_without_improvement: int = 200,
) -> None:

    model_entry.base_path.mkdir(parents=True, exist_ok=True)

    model = model_entry.model.to(DEVICE)
    forward_criterion = nn.MSELoss()
    forward_optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        forward_optimizer, mode="min", factor=0.5, patience=5
    )

    stats = TrainingStats()
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

        p_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{n_epochs}",
            unit="batch",
        )

        for batch_idx, (batch_parameters, batch_targets) in enumerate(p_bar):
            forward_optimizer.zero_grad(set_to_none=True)

            prediction = model(batch_parameters).squeeze(-1)
            batch_loss = forward_criterion(prediction, batch_targets)
            batch_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            forward_optimizer.step()

            train_loss += batch_loss.item()
            running_loss = train_loss / (batch_idx + 1)

            p_bar.set_postfix(
                loss=f"{batch_loss.item():.3e}",
                avg=f"{running_loss:.3e}",
                lr=f"{current_lr:.1e}",
            )

        train_loss /= n_batch

        # Validation step
        model.eval()
        validation_loss = 0.0
        with torch.no_grad():
            for val_params, val_targets in val_loader:
                prediction = model(val_params).squeeze(-1)
                validation_loss += forward_criterion(prediction, val_targets).item()

        scheduler.step(validation_loss)

        current_weight_decay = float(
            forward_optimizer.param_groups[0].get("weight_decay", 0.0)
        )
        stats.append(
            train_loss=train_loss,
            val_loss=validation_loss,
            weight_decay=current_weight_decay,
        )
        stats.save(model_entry.stats_path)

        epoch_time = time.perf_counter() - epoch_start
        print(
            f"Epoch {epoch + 1:03d} done | train_loss={train_loss:.6e} | "
            f"val_loss={validation_loss:.6e} | best_val={best_val_loss:.6e} | time={epoch_time:.2f}s"
        )

        if validation_loss < best_val_loss:
            best_val_loss = validation_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), model_entry.best_model_path)
            print("✓ Saved new best model checkpoint.")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= max_epochs_without_improvement:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    # Save artifacts
    stats.save(model_entry.stats_path)
    torch.save(model.state_dict(), model_entry.final_model_path)


def get_target_against_z(
    coordinates: tuple[float, float],
    parameters: tuple[float, float, float],
    z_points: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the ground truth target function f(z) over a z-range for fixed (x, y) and wavevectors."""
    n_pts = len(z_points)
    x_t = torch.full((n_pts,), coordinates[0], device=DEVICE)
    y_t = torch.full((n_pts,), coordinates[1], device=DEVICE)
    kx_t = torch.full((n_pts,), parameters[0], device=DEVICE)
    ky_t = torch.full((n_pts,), parameters[1], device=DEVICE)
    kz_t = torch.full((n_pts,), parameters[2], device=DEVICE)

    # Shape expected by _test_function: (6, N)
    params = torch.stack([x_t, y_t, z_points, kx_t, ky_t, kz_t], dim=0)
    return _test_function(params)


def get_prediction_against_z(
    model: nn.Module,
    coordinates: tuple[float, float],
    parameters: tuple[float, float, float],
    z_points: torch.Tensor,
) -> torch.Tensor:
    """Generate predictions f(z) from a model over a z-range for fixed (x, y) and wavevectors."""
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
        return model(inputs).squeeze(-1)


def compare_models_against_z(
    model_zoo: list[ModelZooEntry],
    coordinates: tuple[float, float] | None = None,
    parameters: tuple[float, float, float] | None = None,
) -> tuple[Figure, Axes]:
    """Plot and compares ground truth target vs predictions from multiple models along the z-axis."""
    sample = _generate_parameters(n_samples=1).squeeze(1).cpu()
    coordinates = coordinates or (sample[0].item(), sample[1].item())
    parameters = parameters or (sample[3].item(), sample[4].item(), sample[5].item())

    delta_z = _BOUNDS["z"][1] - _BOUNDS["z"][0]
    z_points = torch.linspace(
        _BOUNDS["z"][0] - 0.5 * delta_z,
        _BOUNDS["z"][1] + 0.5 * delta_z,
        500,
        device=DEVICE,
    )

    targets_values = get_target_against_z(
        coordinates=coordinates, parameters=parameters, z_points=z_points
    )

    fig, ax = get_figure()

    z_np = z_points.cpu().numpy()
    ax.plot(
        z_np,
        targets_values.cpu().numpy(),
        label="Target (Ground Truth)",
        color="black",
        linewidth=2.0,
    )

    for model in model_zoo:
        predictions = get_prediction_against_z(
            model=model.model,
            coordinates=coordinates,
            parameters=parameters,
            z_points=z_points,
        )
        (line,) = ax.plot(z_np, predictions.cpu().numpy())
        line.set_label(model.name)
        line.set_linestyle("--")

    ax.set_xlabel("z")
    ax.set_ylabel("f(z)")
    ax.set_title("Model Comparison vs z-axis")
    ax.legend()
    ax.set_xlim(z_np[0], z_np[-1])  # cspell: disable-line

    ax.axvline(x=_BOUNDS["z"][0], color="gray", linewidth=2.0)  # cspell: disable-line
    ax.axvline(x=_BOUNDS["z"][1], color="gray", linewidth=2.0)  # cspell: disable-line
    return fig, ax


if __name__ == "__main__":
    model_zoo: list[ModelZooEntry] = [
        ModelZooEntry(
            name="PureSIREN",
            train=False,
            base_path=Path("data/example_network"),
            model=PureSIREN(
                param_dim=6, output_dim=1, first_omega_0=1.0, hidden_omega_0=1.0
            ),
        ).load_best(device=DEVICE),
        ModelZooEntry(
            name="CondSIREN",
            train=False,
            base_path=Path("data/example_network"),
            model=ForwardCondSIRENStateModel(
                param_dim=6, output_dim=1, first_omega_0=1.0, hidden_omega_0=1.0
            ),
        ).load_best(device=DEVICE),
        ModelZooEntry(
            name="PlainMLP",
            train=False,
            base_path=Path("data/example_network"),
            model=PureMLP(param_dim=6, output_dim=1),
        ).load_best(device=DEVICE),
        ModelZooEntry(
            name="ExplicitAsymptoticGaborNet",
            train=False,
            base_path=Path("data/example_network"),
            model=ExplicitAsymptoticGaborNet(
                in_dim=6,
                hidden_dim=128,
                omega_0=3.0,
                sigma_0=1.5,
                output_dim=1,
            ),
        ).load_best(device=DEVICE),
        ModelZooEntry(
            name="ExplicitAsymptoticGaborNet1",
            train=False,
            base_path=Path("data/example_network"),
            model=ExplicitAsymptoticGaborNet1(
                in_dim=6,
                hidden_dim=128,
                omega_0=3.0,
                output_dim=1,
            ),
        ).load_best(device=DEVICE),
    ]

    for m in model_zoo:
        if m.train:
            print(f"\n[{m.name}] Training started on device: {DEVICE}")
            train_model(model_entry=m, epochs=(200, 100, 100))

    for model in model_zoo:
        if model.stats_path.exists():
            stats = TrainingStats.load(model.stats_path)
            fig, ax = plot_loss_curves(stats)
            fig.savefig(model.base_path / model.name / "loss_curves.pdf")

    fig, _ = compare_model_validation_loss(model_zoo=model_zoo)
    fig.savefig("data/example_network/model_validation_loss_comparison.pdf")
    fig, _ = compare_models_against_z(model_zoo=model_zoo)
    fig.savefig("data/example_network/model_comparison_vs_z.pdf")
