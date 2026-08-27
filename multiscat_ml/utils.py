"""Utility functions and classes for multiscat_ml."""

import json
import pickle  # ruff: ignore[suspicious-pickle-import]
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from slate_core.plot import get_figure


@dataclass
class TrainingStats:
    """Dataclass to store and append statistics from a model training run.

    Attributes
    ----------
    train_loss : list[float]
        History of training loss values per epoch.
    val_loss : list[float]
        History of validation loss values per epoch.
    weight_decay : list[float]
        History of weight decay values per epoch.
    """

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    weight_decay: list[float] = field(default_factory=list)

    def append(
        self,
        train_loss: float,
        val_loss: float,
        weight_decay: float = 0.0,
    ) -> None:
        """Append metrics from a completed epoch.

        Parameters
        ----------
        train_loss : float
            Training loss for the epoch.
        val_loss : float
            Validation loss for the epoch.
        weight_decay : float, optional
            Weight decay value for the epoch, by default 0.0.
        """
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)
        self.weight_decay.append(weight_decay)

    def save(self, path: Path | str) -> None:
        """Save training statistics to a pickle file.

        Parameters
        ----------
        path : Path | str
            File path where pickle data will be written.
        """
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        with path_obj.open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path | str):  # ruff: ignore[missing-return-type-class-method]
        """Load training statistics from a pickle file.

        Parameters
        ----------
        path : Path | str
            File path from which to read pickle data.

        Returns
        -------
        TrainingStats
            Instantiated TrainingStats object.
        """
        path_obj = Path(path)
        with path_obj.open("rb") as f:
            return pickle.load(f)  # ruff: ignore[suspicious-pickle-usage]


def plot_validation_loss(
    stats: TrainingStats | Sequence[float],
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, Line2D]:
    """Plot validation loss from training stats or a sequence of loss values.

    Parameters
    ----------
    stats : TrainingStats | Sequence[float]
        The training statistics or sequence of validation loss values.
    ax : Axes | None, optional
        Matplotlib Axes to plot on. If None, a new figure and axes will be created using get_figure.

    Returns
    -------
    tuple[Figure, Axes, Line2D]
        The figure, axes, and line object.
    """
    fig, ax_res = get_figure(ax)
    values = stats.val_loss if isinstance(stats, TrainingStats) else list(stats)
    epochs = list(range(1, len(values) + 1))
    (line,) = ax_res.plot(epochs, values, label="Validation Loss")
    return fig, ax_res, line


def plot_training_loss(
    stats: TrainingStats | Sequence[float],
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, Line2D]:
    """Plot training loss from training stats or a sequence of loss values.

    Parameters
    ----------
    stats : TrainingStats | Sequence[float]
        The training statistics or sequence of training loss values.
    ax : Axes | None, optional
        Matplotlib Axes to plot on. If None, a new figure and axes will be created using get_figure.

    Returns
    -------
    tuple[Figure, Axes, Line2D]
        The figure, axes, and line object.
    """
    fig, ax_res = get_figure(ax)
    values = stats.train_loss if isinstance(stats, TrainingStats) else list(stats)
    epochs = list(range(1, len(values) + 1))
    (line,) = ax_res.plot(epochs, values, label="Training Loss")
    return fig, ax_res, line


def plot_weight_decay(
    stats: TrainingStats | Sequence[float],
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, Line2D]:
    """Plot weight decay from training stats or a sequence of weight decay values.

    Parameters
    ----------
    stats : TrainingStats | Sequence[float]
        The training statistics or sequence of weight decay values.
    ax : Axes | None, optional
        Matplotlib Axes to plot on. If None, a new figure and axes will be created using get_figure.

    Returns
    -------
    tuple[Figure, Axes, Line2D]
        The figure, axes, and line object.
    """
    fig, ax_res = get_figure(ax)
    values = stats.weight_decay if isinstance(stats, TrainingStats) else list(stats)
    epochs = list(range(1, len(values) + 1))
    (line,) = ax_res.plot(epochs, values, label="Weight Decay")
    return fig, ax_res, line


def plot_loss_curves(
    stats: TrainingStats,
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot validation and training loss curves for a model.

    Parameters
    ----------
    stats : TrainingStats
        Training statistics containing train and validation loss histories.
    ax : Axes | None, optional
        Optional Matplotlib Axes to plot on.

    Returns
    -------
    tuple[Figure, Axes]
        The figure and axes containing the plotted loss curves.
    """
    fig, ax_res = get_figure(ax)

    plot_training_loss(stats, ax=ax_res)
    plot_validation_loss(stats, ax=ax_res)

    ax_res.set_yscale("log")
    ax_res.set_xlabel("Epochs")
    ax_res.set_ylabel("Loss (Log Scale)")
    ax_res.set_title("Training and Validation Loss")
    ax_res.legend()
    return fig, ax_res


def save_loss_history(loss_history: dict, path: str | Path) -> None:
    """Save the loss history as a json file."""
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(loss_history, f, indent=4)


def plot_training_convergence(history: dict, save_path: Path) -> None:
    """Generate a publication-grade log-scale convergence plot."""
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
