from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

import torch
from slate_core.plot import get_figure

from multiscat_ml import TrainingStats, plot_validation_loss

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from torch import nn


@dataclass(kw_only=True, frozen=True)
class ModelZooEntry:
    """Entry in model_zoo containing training/loading flags, base path, and PyTorch model."""

    train: bool = False
    base_path: Path
    model: nn.Module
    name: str

    @property
    def stats_path(self) -> Path:
        """Path to the training statistics pickle file."""
        return self.base_path / self.name / "training_stats.pkl"

    @property
    def best_model_path(self) -> Path:
        """Path to the best model checkpoint file."""
        return self.base_path / self.name / "best_model.pth"

    @property
    def final_model_path(self) -> Path:
        """Path to the final model checkpoint file."""
        return self.base_path / self.name / "final_model.pth"

    def load_best(self, device: torch.device) -> Self:
        """Load the model weights from the best checkpoint if it exists."""
        if self.best_model_path.exists():
            self.model.load_state_dict(
                torch.load(self.best_model_path, map_location=device)
            )

        return self


def compare_model_validation_loss(
    model_zoo: list[ModelZooEntry],
) -> tuple[Figure, Axes]:
    """Compare the validation loss of multiple models in the model zoo."""
    fig, ax = get_figure()
    for model in model_zoo:
        if model.stats_path.exists():
            stats = TrainingStats.load(model.stats_path)
            fig, ax, line = plot_validation_loss(stats, ax=ax)
            line.set_label(model.name)

    ax.set_yscale("log")
    ax.set_xlabel("Epochs", fontsize=12, fontweight="bold")
    ax.set_ylabel("Validation Loss (Log Scale)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Model Validation Loss Comparison",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(frameon=True, facecolor="white", edgecolor="none")

    return fig, ax
