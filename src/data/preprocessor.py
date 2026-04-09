import gc
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from typing import Dict, Tuple

from .data_utils import (
    url_to_tensor,
    url_to_feature_tensor,
    extract_html_features,
    clean_html_text,
    get_image_transforms,
)

logger = logging.getLogger(__name__)


def _precompute_text(df: pd.DataFrame, tokenizer, config: dict):
    """
    Pre-clean HTML and tokenize ALL text once upfront.
    Returns lists of (input_ids, attention_mask) tensors and url tensors.
    """
    max_text_len = config["text"]["max_length"]
    max_url_len = config["url"]["max_length"]

    logger.info("Pre-processing text and URLs (one-time)...")

    # Clean all HTML -> plain text in bulk
    texts = []
    for html in df["html_content"]:
        text = clean_html_text(str(html)) if pd.notna(html) else ""
        texts.append(text if text else "[PAD]")

    # Batch tokenize (much faster than one-by-one)
    encodings = tokenizer(
        texts,
        max_length=max_text_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    # Pre-compute URL tensors
    url_tensors = []
    for url in df["url"]:
        url_str = str(url) if pd.notna(url) else ""
        url_tensors.append(url_to_tensor(url_str, max_url_len))
    url_tensors = torch.stack(url_tensors)

    logger.info("Pre-processing complete.")
    return encodings["input_ids"], encodings["attention_mask"], url_tensors


class PhishingMultiModalDataset(Dataset):
    """
    Dataset with pre-computed text/URL tensors.
    Only images are loaded on-the-fly (they're large, can't all fit in RAM).
    """

    def __init__(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        url_tensors: torch.Tensor,
        image_paths: list,
        labels: torch.Tensor,
        image_size: int,
        augment: bool = False,
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.url_tensors = url_tensors
        self.image_paths = image_paths
        self.labels = labels
        self.transform = get_image_transforms(image_size, augment=augment)
        self.image_size = image_size

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Image (loaded on the fly)
        try:
            img = Image.open(self.image_paths[idx]).convert("RGB")
            image_tensor = self.transform(img)
        except Exception:
            image_tensor = torch.zeros(3, self.image_size, self.image_size)

        return {
            "url": self.url_tensors[idx],
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "image": image_tensor,
            "label": self.labels[idx],
        }


def _make_dataset(df, tokenizer, config, split_name):
    """Build a dataset for one split with pre-computed tensors."""
    df = df.reset_index(drop=True)
    input_ids, attention_mask, url_tensors = _precompute_text(df, tokenizer, config)
    labels = torch.tensor(df["label"].values, dtype=torch.long)
    image_paths = df["image_path"].tolist()

    return PhishingMultiModalDataset(
        input_ids=input_ids,
        attention_mask=attention_mask,
        url_tensors=url_tensors,
        image_paths=image_paths,
        labels=labels,
        image_size=config["visual"]["image_size"],
        augment=(split_name == "train"),
    )


def create_dataloaders(
    df: pd.DataFrame,
    tokenizer,
    config: dict,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create dataloaders with pre-computed text tensors."""
    bs = config["training"]["batch_size"]

    if "split" in df.columns:
        train_df = df[df["split"] == "train"]
        val_df = df[df["split"] == "val"]
        test_df = df[df["split"] == "test"]
    else:
        train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df["label"], random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42)

    logger.info(f"Split -> train: {len(train_df)} | val: {len(val_df)} | test: {len(test_df)}")

    train_ds = _make_dataset(train_df, tokenizer, config, "train")
    val_ds = _make_dataset(val_df, tokenizer, config, "val")
    test_ds = _make_dataset(test_df, tokenizer, config, "test")

    loader_kwargs = dict(num_workers=0, pin_memory=False)

    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True, **loader_kwargs),
        DataLoader(val_ds, batch_size=bs, shuffle=False, **loader_kwargs),
        DataLoader(test_ds, batch_size=bs, shuffle=False, **loader_kwargs),
    )


def compute_class_weights(df: pd.DataFrame, device: torch.device) -> torch.Tensor:
    train_labels = df[df["split"] == "train"]["label"].values if "split" in df.columns else df["label"].values
    weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=train_labels)
    return torch.tensor(weights, dtype=torch.float32).to(device)


# Fast Pipeline: Pre-extracted Embeddings


def _extract_text_embeddings(df: pd.DataFrame, config: dict) -> torch.Tensor:
    """Extract DistilBERT [CLS] embeddings once for all samples."""
    text_cfg = config["text"]
    logger.info("  Loading DistilBERT tokenizer + model...")
    tokenizer = AutoTokenizer.from_pretrained(text_cfg["model_name"])
    model = AutoModel.from_pretrained(text_cfg["model_name"])
    model.eval()

    # Clean HTML → plain text → tokenize
    texts = []
    for html in df["html_content"]:
        text = clean_html_text(str(html)) if pd.notna(html) else ""
        texts.append(text if text else "[PAD]")

    encodings = tokenizer(
        texts,
        max_length=text_cfg["max_length"],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    # Extract mean-pooled embeddings in batches (better than [CLS] alone)
    batch_size = 32
    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Text features"):
            ids = encodings["input_ids"][i : i + batch_size]
            mask = encodings["attention_mask"][i : i + batch_size]
            out = model(input_ids=ids, attention_mask=mask)
            hidden = out.last_hidden_state                    # (B, L, 768)
            mask_expanded = mask.unsqueeze(-1).float()        # (B, L, 1)
            summed = (hidden * mask_expanded).sum(dim=1)      # (B, 768)
            counts = mask_expanded.sum(dim=1).clamp(min=1)    # (B, 1)
            embeddings.append((summed / counts).cpu())        # mean pool over non-padding

    del model, tokenizer, encodings
    gc.collect()

    return torch.cat(embeddings, dim=0)  # (N, 768)


def _extract_visual_embeddings(df: pd.DataFrame, config: dict) -> torch.Tensor:
    """Extract EfficientNet pooled features once for all samples."""
    visual_cfg = config["visual"]
    image_size = visual_cfg["image_size"]
    transform = get_image_transforms(image_size, augment=False)

    logger.info("  Loading EfficientNet-B0...")
    backbone = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    backbone.classifier = torch.nn.Identity()  # strip classifier → output 1280-d
    backbone.eval()

    batch_size = 16
    embeddings = []
    image_paths = df["image_path"].tolist()

    with torch.no_grad():
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Visual features"):
            batch_imgs = []
            for path in image_paths[i : i + batch_size]:
                try:
                    img = Image.open(path).convert("RGB")
                    batch_imgs.append(transform(img))
                except Exception:
                    batch_imgs.append(torch.zeros(3, image_size, image_size))
            batch_tensor = torch.stack(batch_imgs)
            feats = backbone(batch_tensor)  # (B, 1280)
            embeddings.append(feats.cpu())

    del backbone
    gc.collect()

    return torch.cat(embeddings, dim=0)  # (N, 1280)


def extract_and_save_features(df: pd.DataFrame, config: dict) -> dict:
    """
    One-time extraction of frozen DistilBERT and EfficientNet embeddings.
    Cached to disk — subsequent runs load instantly.
    """
    processed_dir = Path(config["data"]["processed_dir"])
    features_path = processed_dir / "features.pt"

    if features_path.exists():
        logger.info(f"Loading cached features from {features_path}")
        return torch.load(features_path, weights_only=False)

    logger.info("Extracting frozen model embeddings (one-time)...")

    text_emb = _extract_text_embeddings(df, config)
    logger.info(f"  Text embeddings: {text_emb.shape}")

    visual_emb = _extract_visual_embeddings(df, config)
    logger.info(f"  Visual embeddings: {visual_emb.shape}")

    # URL tensors (character-level) + hand-crafted features
    max_url_len = config["url"]["max_length"]
    url_tensors = []
    url_features = []
    for url in df["url"]:
        url_str = str(url) if pd.notna(url) else ""
        url_tensors.append(url_to_tensor(url_str, max_url_len))
        url_features.append(url_to_feature_tensor(url_str))
    url_tensors = torch.stack(url_tensors)
    url_features = torch.stack(url_features)   # (N, 9)

    # HTML structural features (forms, inputs, password fields, etc.)
    logger.info("  Extracting HTML structural features...")
    html_features = []
    for html in df["html_content"]:
        html_features.append(extract_html_features(html))
    html_features = torch.stack(html_features)  # (N, 8)
    logger.info(f"  HTML features: {html_features.shape}")

    labels = torch.tensor(df["label"].values, dtype=torch.long)

    features = {
        "text_embeddings": text_emb,
        "visual_embeddings": visual_emb,
        "url_tensors": url_tensors,
        "url_features": url_features,
        "html_features": html_features,
        "labels": labels,
    }

    torch.save(features, features_path)
    size_mb = features_path.stat().st_size / 1e6
    logger.info(f"Saved features to {features_path} ({size_mb:.1f} MB)")
    return features


class PreExtractedDataset(Dataset):
    """Lightweight dataset: pre-extracted embeddings only. No image I/O or model forward passes."""

    def __init__(self, url_tensors, url_features, html_features, text_embeddings, visual_embeddings, labels):
        self.url_tensors = url_tensors
        self.url_features = url_features
        self.html_features = html_features
        self.text_embeddings = text_embeddings
        self.visual_embeddings = visual_embeddings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "url": self.url_tensors[idx],
            "url_features": self.url_features[idx],
            "html_features": self.html_features[idx],
            "text_emb": self.text_embeddings[idx],
            "visual_emb": self.visual_embeddings[idx],
            "label": self.labels[idx],
        }


def create_fast_dataloaders(
    df: pd.DataFrame, features: dict, config: dict
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create dataloaders from pre-extracted features. Zero image loading during training."""
    bs = config["training"]["batch_size"]

    if "split" not in df.columns:
        raise ValueError("Pre-extracted features require 'split' column in dataframe")

    train_mask = (df["split"] == "train").values
    val_mask = (df["split"] == "val").values
    test_mask = (df["split"] == "test").values

    def _make(mask, shuffle):
        ds = PreExtractedDataset(
            url_tensors=features["url_tensors"][mask],
            url_features=features["url_features"][mask],
            html_features=features["html_features"][mask],
            text_embeddings=features["text_embeddings"][mask],
            visual_embeddings=features["visual_embeddings"][mask],
            labels=features["labels"][mask],
        )
        return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=0, pin_memory=False)

    logger.info(f"Fast splits -> train: {train_mask.sum()} | val: {val_mask.sum()} | test: {test_mask.sum()}")

    return _make(train_mask, True), _make(val_mask, False), _make(test_mask, False)
