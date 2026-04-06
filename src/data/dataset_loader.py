import os
import logging
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)


def _find_label_dirs(base_dir: Path) -> dict:
    """
    Find genuine_site_0 / phishing_site_1 folders,
    handling double-nested Kaggle extractions.
    Returns dict: folder_name -> Path.
    """
    results = {}
    for item in base_dir.iterdir():
        if not item.is_dir():
            continue
        name = item.name.lower()
        if "phishing" in name:
            results["phishing_site_1"] = item
        elif "genuine" in name:
            results["genuine_site_0"] = item
        else:
            for child in item.iterdir():
                if not child.is_dir():
                    continue
                cname = child.name.lower()
                if "phishing" in cname:
                    results["phishing_site_1"] = child
                elif "genuine" in cname:
                    results["genuine_site_0"] = child
    return results


def _build_file_index(folder_path: Path, extensions=None) -> dict:
    """Build mapping: numeric_suffix -> file_path for all files in folder."""
    index_map = {}
    if not folder_path or not folder_path.exists():
        return index_map
    for f in folder_path.iterdir():
        if not f.is_file():
            continue
        if extensions and f.suffix.lower() not in extensions:
            continue
        # Extract numeric suffix: "domain_141" -> 141
        parts = f.stem.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            index_map[int(parts[1])] = str(f)
    return index_map


class PhishingDatasetLoader:
    """
    Matches HuggingFace URL entries to Kaggle HTML/screenshot files
    using the row index → filename suffix mapping.
    Only keeps entries with BOTH HTML and image (inner join).
    Preserves original HF train/val/test splits.
    """

    def __init__(self, config: dict):
        self.raw_dir = Path(config["data"]["raw_dir"])
        self.processed_dir = Path(config["data"]["processed_dir"])
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def build_merged_dataset(self) -> pd.DataFrame:
        cache_path = self.processed_dir / "merged_dataset.csv"
        if cache_path.exists():
            logger.info(f"Loading cached merged dataset from {cache_path}")
            return pd.read_csv(cache_path)

        # ── 1. Load HuggingFace URL data with split info ─────────────────
        urls_dir = self.raw_dir / "urls"
        dfs = []
        for split_name, filename in [
            ("train", "train.csv"),
            ("val", "validation.csv"),
            ("test", "test.csv"),
        ]:
            csv_path = urls_dir / filename
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                df = df.rename(columns={"text": "url", "labels": "label"})
                df["split"] = split_name
                dfs.append(df)
                logger.info(f"  URLs {split_name}: {len(df)} rows")

        if not dfs:
            raise RuntimeError(
                "No URL data found. Run scripts/download_data.py first."
            )

        # Concatenate in split order (train, val, test) and use row index
        df_hf = pd.concat(dfs, ignore_index=True)
        df_hf["filename_index"] = df_hf.index
        df_hf["folder"] = df_hf["label"].apply(
            lambda x: "phishing_site_1" if x == 1 else "genuine_site_0"
        )
        logger.info(f"Total HF entries: {len(df_hf)}")

        # ── 2. Build file indexes for HTML and screenshots ───────────────
        html_dirs = _find_label_dirs(self.raw_dir / "html_content")
        img_dirs = _find_label_dirs(self.raw_dir / "screenshots")

        html_index = {}  # (folder_name, numeric_idx) -> file_path
        img_index = {}

        for folder_name, folder_path in html_dirs.items():
            file_map = _build_file_index(folder_path)
            for idx, path in file_map.items():
                html_index[(folder_name, idx)] = path
            logger.info(f"  HTML {folder_name}: {len(file_map)} files")

        img_exts = {".png", ".jpg", ".jpeg", ".webp"}
        for folder_name, folder_path in img_dirs.items():
            file_map = _build_file_index(folder_path, img_exts)
            for idx, path in file_map.items():
                img_index[(folder_name, idx)] = path
            logger.info(f"  Screenshots {folder_name}: {len(file_map)} files")

        # ── 3. Match each HF entry to HTML + image by index ──────────────
        df_hf["html_path"] = df_hf.apply(
            lambda r: html_index.get((r["folder"], r["filename_index"])),
            axis=1,
        )
        df_hf["image_path"] = df_hf.apply(
            lambda r: img_index.get((r["folder"], r["filename_index"])),
            axis=1,
        )

        has_html = df_hf["html_path"].notna().sum()
        has_img = df_hf["image_path"].notna().sum()
        logger.info(f"Matched HTML: {has_html} | Matched image: {has_img}")

        # ── 4. Keep only entries with BOTH HTML and image ────────────────
        merged = df_hf.dropna(subset=["html_path", "image_path"]).reset_index(
            drop=True
        )
        logger.info(f"After filtering (both HTML + image): {len(merged)} entries")

        # Read HTML content from files (some Kaggle files have Windows-incompatible names)
        def _safe_read(p):
            try:
                return Path(p).read_text(errors="ignore")
            except OSError:
                return ""

        merged["html_content"] = merged["html_path"].apply(_safe_read)
        unreadable = (merged["html_content"] == "").sum()
        if unreadable > 0:
            logger.warning(f"  {unreadable} HTML files could not be read (OS-incompatible filenames)")

        # Keep relevant columns
        merged = merged[
            ["url", "label", "split", "filename_index", "html_content", "image_path"]
        ]

        # ── 5. Summary ──────────────────────────────────────────────────
        for split in ["train", "val", "test"]:
            subset = merged[merged["split"] == split]
            if len(subset) > 0:
                logger.info(
                    f"  {split}: {len(subset)} rows | "
                    f"phishing={(subset['label'] == 1).sum()} | "
                    f"legit={(subset['label'] == 0).sum()}"
                )

        merged.to_csv(cache_path, index=False)
        logger.info(f"Saved merged dataset: {len(merged)} rows -> {cache_path}")
        return merged
