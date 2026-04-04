import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights, EfficientNet_B0_Weights


class VisualEncoder(nn.Module):
    """
    CNN-based webpage screenshot encoder using transfer learning.
    Supports ResNet50 and EfficientNet-B0.
    """

    def __init__(self, config: dict):
        super().__init__()
        visual_cfg = config["visual"]
        self.model_name = visual_cfg["model_name"]
        self.output_dim = visual_cfg["output_dim"]
        self.dropout_p = visual_cfg["dropout"]
        self.freeze_backbone = visual_cfg.get("freeze_layers", True)

        if self.model_name == "resnet50":
            self._build_resnet()
        elif self.model_name == "efficientnet_b0":
            self._build_efficientnet()
        else:
            raise ValueError(f"Unknown visual model: {self.model_name}")

        self.dropout = nn.Dropout(self.dropout_p)
        self.layer_norm = nn.LayerNorm(self.output_dim)

    def _build_resnet(self):
        backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        in_features = backbone.fc.in_features  # 2048

        if self.freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False
            # Unfreeze layer3, layer4, and fc
            for layer in [backbone.layer3, backbone.layer4]:
                for param in layer.parameters():
                    param.requires_grad = True

        backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(512, self.output_dim),
        )
        self.backbone = backbone

    def _build_efficientnet(self):
        backbone = models.efficientnet_b0(
            weights=EfficientNet_B0_Weights.IMAGENET1K_V1
        )
        in_features = backbone.classifier[1].in_features  # 1280

        if self.freeze_backbone:
            for param in backbone.features.parameters():
                param.requires_grad = False
            # Unfreeze last two feature blocks
            for block in list(backbone.features.children())[-2:]:
                for param in block.parameters():
                    param.requires_grad = True

        backbone.classifier = nn.Sequential(
            nn.Dropout(self.dropout_p),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(512, self.output_dim),
        )
        self.backbone = backbone

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (batch, 3, H, W) float tensor
        Returns:
            embedding: (batch, output_dim)
        """
        out = self.backbone(images)
        out = self.dropout(out)
        out = self.layer_norm(out)
        return out
