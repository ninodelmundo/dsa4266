#!/usr/bin/env python3
"""
Preprocess raw data and build the merged dataset CSV.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.helpers import load_config, setup_logging
from src.data.dataset_loader import PhishingDatasetLoader


def main():
    config = load_config()
    logger = setup_logging(config["project"]["output_dir"])

    loader = PhishingDatasetLoader(config)
    merged = loader.build_merged_dataset()

    logger.info(f"Merged dataset shape: {merged.shape}")
    logger.info(f"Label distribution:\n{merged['label'].value_counts()}")
    logger.info(
        f"HTML available: {merged['html_content'].notna().sum()} / {len(merged)}"
    )
    logger.info(
        f"Images available: {merged['image_path'].notna().sum()} / {len(merged)}"
    )


if __name__ == "__main__":
    main()
