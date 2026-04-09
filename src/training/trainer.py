import logging
import time
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
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model_type = model_type

        train_cfg = config["training"]
        self.num_epochs = train_cfg["num_epochs"]
        self.gradient_clip = train_cfg["gradient_clip"]
        self.threshold = train_cfg["decision_threshold"]

        # Loss function
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        else:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # Optimizer — use different LR for BERT layers
        self.optimizer = self._build_optimizer(train_cfg)

        # Scheduler — cosine annealing gives smoother LR decay than ReduceLROnPlateau
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.num_epochs, eta_min=1e-6
        )

        # Callbacks
        self.early_stopping = EarlyStopping(
            patience=train_cfg["patience"], mode="min"
        )
        self.metric_logger = MetricLogger()

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

        return torch.optim.AdamW(
            param_groups, weight_decay=float(train_cfg["weight_decay"])
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

    def _mixup_forward(self, batch, labels):
        """Forward pass with manifold mixup for fast_multimodal."""
        lam = np.random.beta(0.2, 0.2)
        index = torch.randperm(labels.size(0)).to(self.device)
        logits = self.model(
            url_tokens=batch["url"].to(self.device),
            url_features=batch["url_features"].to(self.device),
            html_features=batch["html_features"].to(self.device),
            text_emb=batch["text_emb"].to(self.device),
            visual_emb=batch["visual_emb"].to(self.device),
            mixup_lambda=lam,
            mixup_index=index,
        )
        loss = lam * self.criterion(logits, labels) + (1 - lam) * self.criterion(logits, labels[index])
        return logits, loss

    def train_epoch(self) -> Tuple[float, Dict]:
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        all_labels, all_probs = [], []

        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for batch in pbar:
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()

            # Use manifold mixup for fast_multimodal during training
            if self.model_type == "fast_multimodal":
                logits, loss = self._mixup_forward(batch, labels)
            else:
                logits = self._forward_batch(batch)
                loss = self.criterion(logits, labels)

            loss.backward()

            # Gradient clipping
            if self.gradient_clip > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip
                )

            self.optimizer.step()

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
        )
        return avg_loss, metrics

    def fit(self, output_dir: str) -> Dict[str, List]:
        """Full training loop with early stopping."""
        history = {
            "train_loss": [],
            "val_loss": [],
            "train_f1": [],
            "val_f1": [],
        }
        best_val_loss = float("inf")

        for epoch in range(1, self.num_epochs + 1):
            start = time.time()

            train_loss, train_metrics = self.train_epoch()
            val_loss, val_metrics = self.validate()

            self.scheduler.step()

            elapsed = time.time() - start
            logger.info(
                f"Epoch {epoch}/{self.num_epochs} ({elapsed:.1f}s) | "
                f"Train Loss: {train_loss:.4f} F1: {train_metrics['f1']:.4f} | "
                f"Val Loss: {val_loss:.4f} F1: {val_metrics['f1']:.4f} "
                f"ROC-AUC: {val_metrics['roc_auc']:.4f}"
            )

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_f1"].append(train_metrics["f1"])
            history["val_f1"].append(val_metrics["f1"])

            self.metric_logger.log(epoch, train_loss, val_loss, val_metrics)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    val_metrics,
                    f"{output_dir}/best_model.pt",
                )
                logger.info(f"  -> New best model saved (val_loss={val_loss:.4f})")

            # Early stopping
            if self.early_stopping(val_loss):
                logger.info(
                    f"Early stopping triggered at epoch {epoch}."
                )
                break

        return history
