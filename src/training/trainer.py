import copy
import logging
import os
import time
from contextlib import nullcontext
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

from ..evaluation.metrics import compute_metrics
from ..utils.helpers import save_checkpoint
from .callbacks import EarlyStopping, MetricLogger

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """
    Focal Loss: down-weights easy examples so training focuses on hard
    boundary cases.  FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.weight = weight          # per-class weight tensor
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        # Apply label smoothing via standard CE first
        ce = nn.functional.cross_entropy(
            logits, targets,
            weight=self.weight,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce)           # p_t = probability of correct class
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean()


class Trainer:
    """
    Generic trainer that works for both unimodal and multimodal models.
    """

    def __init__(
        self,
        model: nn.Module,
        config: dict,
        device: torch.device,
        train_loader: DataLoader,
        val_loader: DataLoader,
        class_weights: Optional[torch.Tensor] = None,
        model_type: str = "multimodal",
        metric_weights: Optional[Dict[str, float]] = None,
        checkpoint_metric: str = "val_loss",
        use_amp: bool = False,
        trial=None,
        trial_report_metric: str = "composite_score",
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model_type = model_type
        self.metric_weights = metric_weights
        self.checkpoint_metric = checkpoint_metric
        self.use_amp = bool(use_amp and device.type == "cuda")
        self.trial = trial
        self.trial_report_metric = trial_report_metric

        train_cfg = config["training"]
        self.num_epochs = train_cfg["num_epochs"]
        self.gradient_clip = train_cfg["gradient_clip"]
        self.threshold = train_cfg["decision_threshold"]
        self.scheduler_name = train_cfg.get("scheduler", "plateau").lower()
        self.amp_dtype = config.get("runtime", {}).get("autocast_dtype", "float16")
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.use_amp,
        )

        # Loss function
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        else:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # Optimizer — use different LR for BERT layers
        self.optimizer = self._build_optimizer(train_cfg)

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=3
        ) if self.scheduler_name == "plateau" else self._build_scheduler(train_cfg)

        # Callbacks
        self.early_stopping = EarlyStopping(
            patience=train_cfg["patience"], mode="min"
        )
        self.metric_logger = MetricLogger()
        self.best_checkpoint_value = None

    def _resolve_amp_dtype(self) -> torch.dtype:
        if self.amp_dtype == "bfloat16":
            return torch.bfloat16
        return torch.float16

    def _autocast_context(self):
        if not self.use_amp:
            return nullcontext()
        return torch.autocast(
            device_type=self.device.type,
            dtype=self._resolve_amp_dtype(),
        )

    def _build_optimizer(self, train_cfg: dict) -> torch.optim.Optimizer:
        """Create optimizer with per-group learning rates."""
        bert_params = []
        other_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "bert" in name:
                bert_params.append(param)
            else:
                other_params.append(param)

        param_groups = [
            {"params": other_params, "lr": float(train_cfg["learning_rate"])},
        ]
        if bert_params:
            param_groups.append(
                {"params": bert_params, "lr": float(train_cfg["bert_learning_rate"])}
            )

        optimizer_name = train_cfg.get("optimizer", "adamw").lower()
        weight_decay = float(train_cfg["weight_decay"])
        if optimizer_name == "rmsprop":
            return torch.optim.RMSprop(
                param_groups,
                weight_decay=weight_decay,
                momentum=float(train_cfg.get("rmsprop_momentum", 0.0)),
            )
        return torch.optim.AdamW(param_groups, weight_decay=weight_decay)

    def _build_scheduler(self, train_cfg: dict):
        scheduler_name = train_cfg.get("scheduler", "plateau").lower()
        if scheduler_name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=max(1, self.num_epochs),
                eta_min=float(train_cfg.get("min_learning_rate", 1e-6)),
            )
        if scheduler_name == "onecycle":
            max_lrs = [group["lr"] for group in self.optimizer.param_groups]
            return torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=max_lrs,
                epochs=self.num_epochs,
                steps_per_epoch=max(1, len(self.train_loader)),
                pct_start=float(train_cfg.get("onecycle_pct_start", 0.3)),
            )
        if scheduler_name == "none":
            return None
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=3
        )

    def _forward_batch(self, batch: dict) -> torch.Tensor:
        """Run a forward pass for any model type."""
        if self.model_type == "multimodal":
            return self.model(
                url_tokens=batch["url"].to(self.device),
                input_ids=batch["input_ids"].to(self.device),
                attention_mask=batch["attention_mask"].to(self.device),
                images=batch["image"].to(self.device),
            )
        elif self.model_type == "url":
            return self.model(url_tokens=batch["url"].to(self.device))
        elif self.model_type == "text":
            return self.model(
                input_ids=batch["input_ids"].to(self.device),
                attention_mask=batch["attention_mask"].to(self.device),
            )
        elif self.model_type == "visual":
            return self.model(images=batch["image"].to(self.device))
        elif self.model_type == "fast_multimodal":
            return self.model(
                url_tokens=batch["url"].to(self.device),
                url_features=batch["url_features"].to(self.device),
                html_features=batch["html_features"].to(self.device),
                text_emb=batch["text_emb"].to(self.device),
                visual_emb=batch["visual_emb"].to(self.device),
            )
        elif self.model_type == "fast_text":
            return self.model(text_emb=batch["text_emb"].to(self.device))
        elif self.model_type == "fast_visual":
            return self.model(visual_emb=batch["visual_emb"].to(self.device))
        elif self.model_type == "fast_url":
            return self.model(
                url_tokens=batch["url"].to(self.device),
                url_features=batch["url_features"].to(self.device),
            )
        elif self.model_type == "fast_html":
            return self.model(html_features=batch["html_features"].to(self.device))

    def _checkpoint_value(self, val_loss: float, val_metrics: Dict[str, float]) -> float:
        if self.checkpoint_metric == "composite_score":
            return float(val_metrics.get("composite_score", float("-inf")))
        return -float(val_loss)

    def train_epoch(self) -> Tuple[float, Dict]:
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        all_labels, all_probs = [], []

        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for batch in pbar:
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()
            with self._autocast_context():
                logits = self._forward_batch(batch)
                loss = self.criterion(logits, labels)

            if self.use_amp:
                self.scaler.scale(loss).backward()
                if self.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.gradient_clip > 0:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip
                    )
                self.optimizer.step()

            if self.scheduler_name == "onecycle" and self.scheduler is not None:
                self.scheduler.step()

            running_loss += loss.item() * labels.size(0)
            probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = running_loss / len(self.train_loader.dataset)
        metrics = compute_metrics(
            np.array(all_labels),
            (np.array(all_probs) >= self.threshold).astype(int),
            np.array(all_probs),
            self.threshold,
            metric_weights=self.metric_weights,
        )
        return avg_loss, metrics

    @torch.no_grad()
    def validate(self) -> Tuple[float, Dict]:
        """Validate on the validation set."""
        self.model.eval()
        running_loss = 0.0
        all_labels, all_probs = [], []

        for batch in tqdm(self.val_loader, desc="Validating", leave=False):
            labels = batch["label"].to(self.device)
            with self._autocast_context():
                logits = self._forward_batch(batch)
                loss = self.criterion(logits, labels)

            running_loss += loss.item() * labels.size(0)
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)

        avg_loss = running_loss / len(self.val_loader.dataset)
        metrics = compute_metrics(
            np.array(all_labels),
            (np.array(all_probs) >= self.threshold).astype(int),
            np.array(all_probs),
            self.threshold,
            metric_weights=self.metric_weights,
        )
        return avg_loss, metrics

    def fit(self, output_dir: str) -> Dict[str, List]:
        """Full training loop with early stopping."""
        history = {
            "train_loss": [],
            "val_loss": [],
            "train_f1": [],
            "val_f1": [],
            "train_composite": [],
            "val_composite": [],
        }
        best_checkpoint_path = os.path.join(output_dir, "best_model.pt")
        best_state_dict = None

        for epoch in range(1, self.num_epochs + 1):
            start = time.time()

            train_loss, train_metrics = self.train_epoch()
            val_loss, val_metrics = self.validate()

            if self.scheduler_name == "plateau" and self.scheduler is not None:
                self.scheduler.step(val_loss)
            elif self.scheduler_name == "cosine" and self.scheduler is not None:
                self.scheduler.step()

            elapsed = time.time() - start
            logger.info(
                f"Epoch {epoch}/{self.num_epochs} ({elapsed:.1f}s) | "
                f"Train Loss: {train_loss:.4f} F1: {train_metrics['f1']:.4f} | "
                f"Val Loss: {val_loss:.4f} F1: {val_metrics['f1']:.4f} "
                f"ROC-AUC: {val_metrics['roc_auc']:.4f} "
                f"C-Index: {val_metrics['c_index']:.4f}"
            )

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_f1"].append(train_metrics["f1"])
            history["val_f1"].append(val_metrics["f1"])
            history["train_composite"].append(
                train_metrics.get("composite_score", float("nan"))
            )
            history["val_composite"].append(
                val_metrics.get("composite_score", float("nan"))
            )

            self.metric_logger.log(epoch, train_loss, val_loss, val_metrics)

            # Save best model
            checkpoint_value = self._checkpoint_value(val_loss, val_metrics)
            if (
                self.best_checkpoint_value is None
                or checkpoint_value > self.best_checkpoint_value
            ):
                self.best_checkpoint_value = checkpoint_value
                best_state_dict = copy.deepcopy(self.model.state_dict())
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    val_metrics,
                    best_checkpoint_path,
                )
                if self.checkpoint_metric == "composite_score":
                    logger.info(
                        "  -> New best model saved "
                        f"(composite_score={val_metrics.get('composite_score', float('nan')):.4f})"
                    )
                else:
                    logger.info(f"  -> New best model saved (val_loss={val_loss:.4f})")

            if self.trial is not None:
                report_value = (
                    val_metrics.get(self.trial_report_metric, float("-inf"))
                    if self.trial_report_metric != "val_loss"
                    else -val_loss
                )
                self.trial.report(report_value, step=epoch)
                if self.trial.should_prune():
                    try:
                        import optuna
                    except ImportError:
                        optuna = None
                    if optuna is not None:
                        raise optuna.TrialPruned()

            # Early stopping
            if self.early_stopping(val_loss):
                logger.info(
                    f"Early stopping triggered at epoch {epoch}."
                )
                break

        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)

        return history
