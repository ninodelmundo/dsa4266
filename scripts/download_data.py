#!/usr/bin/env python3
"""
Download datasets from Kaggle and HuggingFace.

Prerequisites:
  - pip install kaggle datasets python-dotenv
  - Set KAGGLE_USERNAME and KAGGLE_KEY in .env file
"""
import os
import sys
import logging
from pathlib import Path
from datasets import load_dataset
import kagglehub
import shutil

# Load .env before importing kaggle (it reads env vars on import)
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")


def download_kaggle_dataset(dataset_slug: str, target_dir: Path):
    """Download a Kaggle dataset using kagglehub and move it to target_dir."""
    try:

        logger.info(f"Downloading {dataset_slug} using kagglehub...")

        # Download dataset (returns cache path)
        downloaded_path = kagglehub.dataset_download(dataset_slug)

        logger.info(f"Downloaded to cache: {downloaded_path}")

        # Ensure target directory exists
        target_dir.mkdir(parents=True, exist_ok=True)

        # Copy contents from kagglehub cache → target_dir
        for item in Path(downloaded_path).iterdir():
            dest = target_dir / item.name

            if item.is_dir():
                shutil.copytree(item, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dest)

        logger.info(f"Dataset copied to: {target_dir}")

    except Exception as e:
        logger.error(f"Could not download {dataset_slug}: {e}")


def download_huggingface_dataset():
    """Download URL dataset from HuggingFace — all splits."""
    try:

        logger.info("Downloading URL dataset from HuggingFace...")
        ds = load_dataset("abhinavsarkar/phising-site-datasets")
        cache_dir = RAW_DIR / "urls"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Save each split (train, validation, test)
        for split_name, split_data in ds.items():
            df = split_data.to_pandas()
            df.to_csv(cache_dir / f"{split_name}.csv", index=False)
            logger.info(f"  {split_name}: {len(df)} rows -> {cache_dir / f'{split_name}.csv'}")

        # Also save a combined version
        import pandas as pd
        all_dfs = [split.to_pandas() for split in ds.values()]
        combined = pd.concat(all_dfs, ignore_index=True)
        combined.to_csv(cache_dir / "all_urls.csv", index=False)
        logger.info(f"  Combined: {len(combined)} rows -> {cache_dir / 'all_urls.csv'}")

    except Exception as e:
        logger.error(f"Could not download HuggingFace dataset: {e}")


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # 1. HuggingFace — URLs (small, fast)
    download_huggingface_dataset()

    # 2. Kaggle — screenshots (~207 MB)
    download_kaggle_dataset(
        "zackyzac/phishing-sites-screenshot", RAW_DIR / "screenshots"
    )

    # 3. Kaggle — HTML content (~263 MB)
    download_kaggle_dataset(
        "zackyzac/phishing-site-html-content", RAW_DIR / "html_content"
    )

    logger.info("All downloads complete.")


if __name__ == "__main__":
    main()
