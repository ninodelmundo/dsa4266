import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .url_model import build_url_encoder
from .text_model import TextEncoder
from .visual_model import VisualEncoder
from ..data.data_utils import URL_FEATURES_DIM, HTML_FEATURES_DIM


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

        # Per-modality projection to common dim
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

        # Fusion-specific parameters
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

        # Classifier head
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
        # Encode each modality
        url_emb = self.url_proj(self.url_encoder(url_tokens))       # (B, D)
        text_emb = self.text_proj(
            self.text_encoder(input_ids, attention_mask)
        )                                                            # (B, D)
        visual_emb = self.visual_proj(self.visual_encoder(images))   # (B, D)

        # Fuse
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


# Unimodal Wrappers for single-modality baselines


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


# Fast variants using pre-extracted embeddings


class FastFusionClassifier(nn.Module):
    """
    Fast multi-modal fusion using pre-extracted DistilBERT (768-d) and
    EfficientNet (1280-d) embeddings.  Only the URL encoder, projection
    heads, and classifier run during training.
    """

    def __init__(self, config: dict):
        super().__init__()
        fusion_cfg = config["fusion"]
        self.strategy = fusion_cfg["strategy"]
        self.projected_dim = fusion_cfg["projected_dim"]
        self.hidden_dim = fusion_cfg["hidden_dim"]
        self.dropout_p = fusion_cfg["dropout"]
        self.disabled_modalities = set(fusion_cfg.get("disabled_modalities", []))
        self.use_url_scalar_features = fusion_cfg.get("use_url_scalar_features", True)
        self.active_modalities = [
            name
            for name in ["url", "text", "visual", "html"]
            if name not in self.disabled_modalities
        ]
        if not self.active_modalities:
            raise ValueError("FastFusionClassifier requires at least one active modality")

        # URL encoder runs live (small, fast on CPU)
        self.url_encoder = build_url_encoder(config)

        # Per-modality projections from raw feature dimensions
        # url_proj takes BiLSTM output (url.output_dim) + 9 hand-crafted features
        self.url_proj = nn.Sequential(
            nn.Linear(
                config["url"]["output_dim"] + (URL_FEATURES_DIM if self.use_url_scalar_features else 0),
                self.projected_dim,
            ),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.LayerNorm(self.projected_dim),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(768, self.projected_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.LayerNorm(self.projected_dim),
        )
        self.visual_proj = nn.Sequential(
            nn.Linear(1280, self.projected_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.LayerNorm(self.projected_dim),
        )
        # HTML structural features (forms, inputs, password fields, etc.)
        self.html_proj = nn.Sequential(
            nn.Linear(HTML_FEATURES_DIM, self.projected_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.LayerNorm(self.projected_dim),
        )

        # Fusion-specific parameters
        if self.strategy == "weighted":
            self.modality_weights = nn.Parameter(torch.ones(len(self.active_modalities)))
        elif self.strategy == "attention":
            self.attention = nn.MultiheadAttention(
                embed_dim=self.projected_dim,
                num_heads=fusion_cfg.get("attention_heads", 4),
                dropout=self.dropout_p,
                batch_first=True,
            )
            self.attention_norm = nn.LayerNorm(self.projected_dim)

        # Classifier head
        if self.strategy == "concatenation":
            classifier_in = self.projected_dim * len(self.active_modalities)
        else:
            classifier_in = self.projected_dim

        self.classifier = nn.Sequential(
            nn.Linear(classifier_in, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.hidden_dim, 2),
        )

    def forward(
        self,
        url_tokens: torch.Tensor,
        url_features: torch.Tensor,
        html_features: torch.Tensor,
        text_emb: torch.Tensor,
        visual_emb: torch.Tensor,
        mixup_lambda: float = None,
        mixup_index: torch.Tensor = None,
    ) -> torch.Tensor:
        embeddings = []
        if "url" not in self.disabled_modalities:
            url_encoder_out = self.url_encoder(url_tokens)
            if self.use_url_scalar_features:
                url_encoder_out = torch.cat([url_encoder_out, url_features], dim=-1)
            embeddings.append(self.url_proj(url_encoder_out))
        if "text" not in self.disabled_modalities:
            embeddings.append(self.text_proj(text_emb))
        if "visual" not in self.disabled_modalities:
            embeddings.append(self.visual_proj(visual_emb))
        if "html" not in self.disabled_modalities:
            embeddings.append(self.html_proj(html_features))

        # Manifold mixup: blend projected embeddings to create synthetic samples
        if mixup_lambda is not None and mixup_index is not None:
            embeddings = [
                mixup_lambda * emb + (1 - mixup_lambda) * emb[mixup_index]
                for emb in embeddings
            ]

        if self.strategy == "concatenation":
            fused = torch.cat(embeddings, dim=-1)
        elif self.strategy == "weighted":
            w = F.softmax(self.modality_weights, dim=0)
            fused = sum(weight * emb for weight, emb in zip(w, embeddings))
        elif self.strategy == "attention":
            stack = torch.stack(embeddings, dim=1)
            attended, _ = self.attention(stack, stack, stack)
            attended = self.attention_norm(attended + stack)
            fused = attended.mean(dim=1)

        return self.classifier(fused)

    def get_modality_weights(self) -> Optional[torch.Tensor]:
        if self.strategy == "weighted":
            return F.softmax(self.modality_weights.detach(), dim=0).cpu()
        return None

    def get_active_modalities(self):
        return list(self.active_modalities)


class FastURLOnlyClassifier(nn.Module):
    """URL-only baseline using the fast-pipeline URL features."""

    def __init__(self, config: dict):
        super().__init__()
        url_cfg = config["url"]
        self.use_url_scalar_features = url_cfg.get("use_url_scalar_features", True)
        self.encoder = build_url_encoder(config)
        classifier_in = url_cfg["output_dim"] + (
            URL_FEATURES_DIM if self.use_url_scalar_features else 0
        )
        hidden_dim = url_cfg.get("classifier_hidden_dim", 128)
        bottleneck_dim = url_cfg.get("classifier_bottleneck_dim", 64)
        dropout = config["fusion"]["dropout"]
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, 2),
        )

    def forward(self, url_tokens, url_features=None, **kwargs):
        features = self.encoder(url_tokens)
        if self.use_url_scalar_features and url_features is not None:
            features = torch.cat([features, url_features], dim=-1)
        return self.classifier(features)


class FastTextOnlyClassifier(nn.Module):
    """Text-only baseline using pre-extracted 768-d DistilBERT embeddings."""

    def __init__(self, config: dict):
        super().__init__()
        hidden_dim = config["text"].get("classifier_hidden_dim", 512)
        bottleneck_dim = config["text"].get("classifier_bottleneck_dim", 128)
        self.classifier = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.ReLU(),
            nn.Dropout(config["text"]["dropout"]),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(config["fusion"]["dropout"]),
            nn.Linear(bottleneck_dim, 2),
        )

    def forward(self, text_emb, **kwargs):
        return self.classifier(text_emb)


class FastVisualOnlyClassifier(nn.Module):
    """Visual-only baseline using pre-extracted 1280-d EfficientNet embeddings."""

    def __init__(self, config: dict):
        super().__init__()
        hidden_dim = config["visual"].get("classifier_hidden_dim", 512)
        bottleneck_dim = config["visual"].get("classifier_bottleneck_dim", 128)
        self.classifier = nn.Sequential(
            nn.Linear(1280, hidden_dim),
            nn.ReLU(),
            nn.Dropout(config["visual"]["dropout"]),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(config["fusion"]["dropout"]),
            nn.Linear(bottleneck_dim, 2),
        )

    def forward(self, visual_emb, **kwargs):
        return self.classifier(visual_emb)


class FastHTMLOnlyClassifier(nn.Module):
    """HTML-structure-only baseline using handcrafted HTML features."""

    def __init__(self, config: dict):
        super().__init__()
        hidden_dim = config.get("html", {}).get("classifier_hidden_dim", 128)
        bottleneck_dim = config.get("html", {}).get("classifier_bottleneck_dim", 64)
        dropout = config["fusion"]["dropout"]
        self.classifier = nn.Sequential(
            nn.Linear(HTML_FEATURES_DIM, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, 2),
        )

    def forward(self, html_features, **kwargs):
        return self.classifier(html_features)
