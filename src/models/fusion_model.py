import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .url_model import build_url_encoder
from .text_model import TextEncoder
from .visual_model import VisualEncoder


class FusionClassifier(nn.Module):
    """
    Multi-modal fusion model.
    Combines URL, text, and visual embeddings and classifies phishing vs legitimate.

    Fusion strategies:
      - "concatenation": simple concatenation baseline
      - "weighted":      learnable per-modality scalar weights
      - "attention":     attention-based cross-modal weighting
    """

    def __init__(
        self,
        config: dict,
        url_encoder: Optional[nn.Module] = None,
        text_encoder: Optional[nn.Module] = None,
        visual_encoder: Optional[nn.Module] = None,
    ):
        super().__init__()
        fusion_cfg = config["fusion"]
        self.strategy = fusion_cfg["strategy"]
        self.projected_dim = fusion_cfg["projected_dim"]
        self.hidden_dim = fusion_cfg["hidden_dim"]
        self.dropout_p = fusion_cfg["dropout"]

        # Encoders (build fresh if not supplied)
        self.url_encoder = url_encoder or build_url_encoder(config)
        self.text_encoder = text_encoder or TextEncoder(config)
        self.visual_encoder = visual_encoder or VisualEncoder(config)

        url_dim = config["url"]["output_dim"]
        text_dim = config["text"]["output_dim"]
        visual_dim = config["visual"]["output_dim"]

        # ── Per-modality projection to common dim ─────────────────────────────
        self.url_proj = nn.Sequential(
            nn.Linear(url_dim, self.projected_dim),
            nn.ReLU(),
            nn.LayerNorm(self.projected_dim),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, self.projected_dim),
            nn.ReLU(),
            nn.LayerNorm(self.projected_dim),
        )
        self.visual_proj = nn.Sequential(
            nn.Linear(visual_dim, self.projected_dim),
            nn.ReLU(),
            nn.LayerNorm(self.projected_dim),
        )

        # ── Fusion-specific parameters ────────────────────────────────────────
        if self.strategy == "weighted":
            self.modality_weights = nn.Parameter(torch.ones(3))

        elif self.strategy == "attention":
            self.attention = nn.MultiheadAttention(
                embed_dim=self.projected_dim,
                num_heads=4,
                dropout=self.dropout_p,
                batch_first=True,
            )
            self.attention_norm = nn.LayerNorm(self.projected_dim)

        # ── Classifier head ───────────────────────────────────────────────────
        if self.strategy == "concatenation":
            classifier_in = self.projected_dim * 3
        else:
            classifier_in = self.projected_dim  # weighted sum or attention output

        self.classifier = nn.Sequential(
            nn.Linear(classifier_in, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.hidden_dim // 2, 2),
        )

    def forward(
        self,
        url_tokens: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns:
            logits: (batch, 2)
        """
        # ── Encode each modality ──────────────────────────────────────────────
        url_emb = self.url_proj(self.url_encoder(url_tokens))       # (B, D)
        text_emb = self.text_proj(
            self.text_encoder(input_ids, attention_mask)
        )                                                            # (B, D)
        visual_emb = self.visual_proj(self.visual_encoder(images))   # (B, D)

        # ── Fuse ──────────────────────────────────────────────────────────────
        if self.strategy == "concatenation":
            fused = torch.cat(
                [url_emb, text_emb, visual_emb], dim=-1
            )  # (B, 3D)

        elif self.strategy == "weighted":
            w = F.softmax(self.modality_weights, dim=0)  # normalised weights
            fused = w[0] * url_emb + w[1] * text_emb + w[2] * visual_emb

        elif self.strategy == "attention":
            # Stack modality embeddings as a "sequence"
            stack = torch.stack(
                [url_emb, text_emb, visual_emb], dim=1
            )  # (B, 3, D)
            attended, _ = self.attention(stack, stack, stack)
            attended = self.attention_norm(attended + stack)
            fused = attended.mean(dim=1)  # mean over 3 modalities

        return self.classifier(fused)

    def get_modality_weights(self) -> Optional[torch.Tensor]:
        """Return normalised modality weights (weighted fusion only)."""
        if self.strategy == "weighted":
            return F.softmax(self.modality_weights.detach(), dim=0).cpu()
        return None


# ── Unimodal Wrappers for single-modality baselines ─────────────────────────


class URLOnlyClassifier(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.encoder = build_url_encoder(config)
        self.classifier = nn.Sequential(
            nn.Linear(config["url"]["output_dim"], 128),
            nn.ReLU(),
            nn.Dropout(config["fusion"]["dropout"]),
            nn.Linear(128, 2),
        )

    def forward(self, url_tokens, **kwargs):
        return self.classifier(self.encoder(url_tokens))


class TextOnlyClassifier(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.encoder = TextEncoder(config)
        self.classifier = nn.Sequential(
            nn.Linear(config["text"]["output_dim"], 128),
            nn.ReLU(),
            nn.Dropout(config["fusion"]["dropout"]),
            nn.Linear(128, 2),
        )

    def forward(self, input_ids, attention_mask, **kwargs):
        return self.classifier(self.encoder(input_ids, attention_mask))


class VisualOnlyClassifier(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.encoder = VisualEncoder(config)
        self.classifier = nn.Sequential(
            nn.Linear(config["visual"]["output_dim"], 128),
            nn.ReLU(),
            nn.Dropout(config["fusion"]["dropout"]),
            nn.Linear(128, 2),
        )

    def forward(self, images, **kwargs):
        return self.classifier(self.encoder(images))
