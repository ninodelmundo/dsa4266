import logging
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
from typing import Dict, Tuple

from .data_utils import (
    url_to_tensor,
    clean_html_text,
    get_image_transforms,
)

logger = logging.getLogger(__name__)


class PhishingMultiModalDataset(Dataset):
    """
    PyTorch Dataset yielding (url, text, image, label) for each sample.
    Every entry is guaranteed to have both HTML and image data.
    """

    def __init__(self, dataframe: pd.DataFrame, tokenizer, config: dict, split: str = "train"):
        self.df = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.url_max_len = config["url"]["max_length"]
        self.text_max_len = config["text"]["max_length"]
        self.image_size = config["visual"]["image_size"]
        self.transform = get_image_transforms(self.image_size, augment=(split == "train"))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]

        # URL
        url_str = str(row.get("url", "")) if pd.notna(row.get("url")) else ""
        url_tensor = url_to_tensor(url_str, self.url_max_len)

        # Text (from HTML)
        html = row["html_content"]
        text = clean_html_text(str(html)) if pd.notna(html) else ""
        encoding = self.tokenizer(
            text if text else "[PAD]",
            max_length=self.text_max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Image
        img_path = str(row["image_path"])
        try:
            img = Image.open(img_path).convert("RGB")
            image_tensor = self.transform(img)
        except Exception:
            image_tensor = torch.zeros(3, self.image_size, self.image_size)

        label = torch.tensor(int(row["label"]), dtype=torch.long)

        return {
            "url": url_tensor,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image": image_tensor,
            "label": label,
        }


def create_dataloaders(
    df: pd.DataFrame,
    tokenizer,
    config: dict,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Use the split column from the merged dataset."""
    bs = config["training"]["batch_size"]

    # Use pre-assigned splits from the merged dataset
    if "split" in df.columns:
        train_df = df[df["split"] == "train"]
        val_df = df[df["split"] == "val"]
        test_df = df[df["split"] == "test"]
    else:
        # Fallback: stratified split
        from sklearn.model_selection import train_test_split
        train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df["label"], random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42)

    logger.info(f"Split -> train: {len(train_df)} | val: {len(val_df)} | test: {len(test_df)}")

    train_ds = PhishingMultiModalDataset(train_df, tokenizer, config, "train")
    val_ds = PhishingMultiModalDataset(val_df, tokenizer, config, "val")
    test_ds = PhishingMultiModalDataset(test_df, tokenizer, config, "test")

    loader_kwargs = dict(num_workers=0, pin_memory=False)

    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True, **loader_kwargs),
        DataLoader(val_ds, batch_size=bs, shuffle=False, **loader_kwargs),
        DataLoader(test_ds, batch_size=bs, shuffle=False, **loader_kwargs),
    )


def compute_class_weights(df: pd.DataFrame, device: torch.device) -> torch.Tensor:
    from sklearn.utils.class_weight import compute_class_weight
    weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=df["label"].values)
    return torch.tensor(weights, dtype=torch.float32).to(device)
