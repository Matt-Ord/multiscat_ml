"""Tests for utils module."""

import tempfile
from pathlib import Path

from multiscat_ml.utils import (
    TrainingStats,
)


def test_training_stats() -> None:
    stats = TrainingStats()
    stats.append(train_loss=0.5, val_loss=0.6, weight_decay=0.01)
    stats.append(train_loss=0.3, val_loss=0.4, weight_decay=0.005)

    assert stats.train_loss == [0.5, 0.3]
    assert stats.val_loss == [0.6, 0.4]
    assert stats.weight_decay == [0.01, 0.005]

    # Verify to_dict and from_dict have been removed
    assert not hasattr(stats, "to_dict")
    assert not hasattr(TrainingStats, "from_dict")

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "stats.pkl"
        stats.save(file_path)
        loaded = TrainingStats.load(file_path)

        assert isinstance(loaded, TrainingStats)
        assert loaded.train_loss == [0.5, 0.3]
        assert loaded.val_loss == [0.6, 0.4]
        assert loaded.weight_decay == [0.01, 0.005]
