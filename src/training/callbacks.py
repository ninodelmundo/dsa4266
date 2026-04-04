import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Stop training when a monitored metric stops improving.
    """

    def __init__(self, patience: int = 5, mode: str = "min", min_delta: float = 1e-4):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None

    def __call__(self, metric: float) -> bool:
        if self.best_score is None:
            self.best_score = metric
            return False

        if self.mode == "min":
            improved = metric < (self.best_score - self.min_delta)
        else:
            improved = metric > (self.best_score + self.min_delta)

        if improved:
            self.best_score = metric
            self.counter = 0
        else:
            self.counter += 1
            logger.debug(
                f"EarlyStopping: {self.counter}/{self.patience} "
                f"(best={self.best_score:.4f}, current={metric:.4f})"
            )

        return self.counter >= self.patience


class MetricLogger:
    """
    Accumulates per-epoch metrics for later analysis.
    """

    def __init__(self):
        self.history: List[Dict] = []

    def log(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        val_metrics: Dict[str, float],
    ):
        entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        self.history.append(entry)

    def get_history(self) -> List[Dict]:
        return self.history

    def best_epoch(self, metric: str = "val_f1", mode: str = "max") -> Dict:
        if not self.history:
            return {}
        if mode == "max":
            return max(self.history, key=lambda x: x.get(metric, 0))
        return min(self.history, key=lambda x: x.get(metric, float("inf")))
