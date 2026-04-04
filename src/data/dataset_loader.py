import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


def _find_label_dirs(base_dir: Path):
    """
    Find genuine_site_0 / phishing_site_1 folders,
    handling double-nested Kaggle extractions.
    """
    results = []
    for item in base_dir.iterdir():
        if not item.is_dir():
            continue
        name = item.name.lower()
        if "phishing" in name:
            results.append((item, 1))
        elif "genuine" in name:
            results.append((item, 0))
        else:
            for child in item.iterdir():
                if not child.is_dir():
                    continue
                cname = child.name.lower()
                if "phishing" in cname:
                    results.append((child, 1))
                elif "genuine" in cname:
                    results.append((child, 0))
    return results


class PhishingDatasetLoader:
    """
    Loads HTML + screenshots, inner-joins by filename so every entry
    has BOTH modalities, then optionally matches URLs.
    """

    def __init__(self, config: dict):
        self.raw_dir = Path(config["data"]["raw_dir"])
        self.processed_dir = Path(config["data"]["processed_dir"])
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def load_html(self) -> pd.DataFrame:
        html_dir = self.raw_dir / "html_content"
        if not html_dir.exists():
            return pd.DataFrame(columns=["filename", "html_content", "label"])

        label_dirs = _find_label_dirs(html_dir)
        records = []
        for path, label in label_dirs:
            for f in path.iterdir():
                if f.is_file():
                    try:
                        records.append({
                            "filename": f.stem,
                            "html_content": f.read_text(errors="ignore"),
                            "label": label,
                        })
                    except Exception:
                        pass
            logger.info(f"  HTML {path.name}: {sum(1 for r in records if r['label'] == label)} files")

        return pd.DataFrame(records)

    def load_screenshots(self) -> pd.DataFrame:
        ss_dir = self.raw_dir / "screenshots"
        if not ss_dir.exists():
            return pd.DataFrame(columns=["filename", "image_path", "label"])

        exts = {".png", ".jpg", ".jpeg", ".webp"}
        label_dirs = _find_label_dirs(ss_dir)
        records = []
        for path, label in label_dirs:
            for f in path.iterdir():
                if f.suffix.lower() in exts:
                    records.append({
                        "filename": f.stem,
                        "image_path": str(f),
                        "label": label,
                    })
            logger.info(f"  Screenshots {path.name}: {sum(1 for r in records if r['label'] == label)} files")

        return pd.DataFrame(records)

    def load_urls(self) -> pd.DataFrame:
        """Load all URL CSVs (train/val/test) with split info."""
        urls_dir = self.raw_dir / "urls"
        if not urls_dir.exists():
            return pd.DataFrame(columns=["url", "label", "split"])

        dfs = []
        for split_name, filename in [("train", "train.csv"), ("val", "validation.csv"), ("test", "test.csv")]:
            csv_path = urls_dir / filename
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                # Rename columns: text->url, labels->label
                col_map = {}
                for c in df.columns:
                    if c.lower() in ("text", "url"):
                        col_map[c] = "url"
                    elif c.lower() in ("labels", "label"):
                        col_map[c] = "label"
                df = df.rename(columns=col_map)
                df["split"] = split_name
                dfs.append(df[["url", "label", "split"]])
                logger.info(f"  URLs {split_name}: {len(df)} rows")

        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=["url", "label", "split"])

    def build_merged_dataset(self) -> pd.DataFrame:
        """
        Inner-join HTML + screenshots by filename.
        Every row is guaranteed to have BOTH html_content AND image_path.
        Then match URLs and assign train/val/test splits.
        """
        cache_path = self.processed_dir / "merged_dataset.csv"
        if cache_path.exists():
            logger.info(f"Loading cached merged dataset from {cache_path}")
            return pd.read_csv(cache_path)

        html_df = self.load_html()
        ss_df = self.load_screenshots()
        url_df = self.load_urls()

        logger.info(f"HTML: {len(html_df)} | Screenshots: {len(ss_df)} | URLs: {len(url_df)}")

        if html_df.empty or ss_df.empty:
            raise RuntimeError("Need both HTML and screenshot data. Run scripts/download_data.py first.")

        # Inner join — only keep entries that have BOTH html and image
        merged = html_df.merge(
            ss_df[["filename", "image_path"]],
            on="filename",
            how="inner",
        )
        logger.info(f"After inner join (both HTML + image): {len(merged)} entries")

        # Match URLs by domain
        if not url_df.empty:
            # Build domain -> (url, split) lookup
            url_lookup = {}
            for _, row in url_df.iterrows():
                url = str(row["url"])
                domain = url.replace("https://", "").replace("http://", "").split("/")[0]
                url_lookup[domain] = (url, row["split"])

            domain_col = merged["filename"].astype(str).str.rsplit("_", n=1).str[0]
            merged["url"] = domain_col.map(lambda d: url_lookup.get(d, (None, None))[0])
            merged["split"] = domain_col.map(lambda d: url_lookup.get(d, (None, None))[1])

            matched = merged["url"].notna().sum()
            logger.info(f"URLs matched: {matched}/{len(merged)}")

            # Entries without a split match go to train
            merged["split"] = merged["split"].fillna("train")
        else:
            merged["url"] = None
            merged["split"] = "train"

        # Summary
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
