import re
import string
import numpy as np
import torch
from typing import Optional


# ── URL Utilities ────────────────────────────────────────────────────────────

URL_CHAR_VOCAB = {
    char: idx + 1  # 0 reserved for padding
    for idx, char in enumerate(
        string.ascii_lowercase
        + string.ascii_uppercase
        + string.digits
        + ".-~:/?#[]@!$&'()*+,;=%"
    )
}
URL_VOCAB_SIZE = len(URL_CHAR_VOCAB) + 1  # +1 for padding token


def url_to_tensor(url: str, max_length: int = 200) -> torch.Tensor:
    """Convert a URL string to a character-level integer tensor."""
    url = url.strip().lower()[:max_length]
    indices = [URL_CHAR_VOCAB.get(c, 0) for c in url]
    # Pad or truncate to max_length
    if len(indices) < max_length:
        indices += [0] * (max_length - len(indices))
    return torch.tensor(indices, dtype=torch.long)


URL_FEATURES_DIM = 9


def url_to_feature_tensor(url: str) -> torch.Tensor:
    """
    Extract 9 normalized hand-crafted URL features as a float tensor.
    All values clamped to [0, 1] so they are on the same scale as model activations.
    """
    url = url.strip()
    length       = min(len(url), 500) / 500.0
    num_dots     = min(url.count("."), 20) / 20.0
    num_hyphens  = min(url.count("-"), 20) / 20.0
    num_slashes  = min(url.count("/"), 20) / 20.0
    num_digits   = min(sum(c.isdigit() for c in url), 50) / 50.0
    num_special  = min(sum(c in "@?=&%" for c in url), 20) / 20.0
    has_ip       = float(bool(re.search(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", url)))
    has_https    = float(url.lower().startswith("https"))
    subdomain_count = min(
        len(url.split("/")[2].split(".")) - 2
        if "/" in url and len(url.split("/")) > 2
        else 0,
        10,
    ) / 10.0
    return torch.tensor(
        [length, num_dots, num_hyphens, num_slashes, num_digits,
         num_special, has_ip, has_https, subdomain_count],
        dtype=torch.float32,
    )


def extract_url_features(url: str) -> dict:
    """Extract hand-crafted URL features for analysis."""
    parsed = {}
    parsed["length"] = len(url)
    parsed["num_dots"] = url.count(".")
    parsed["num_hyphens"] = url.count("-")
    parsed["num_slashes"] = url.count("/")
    parsed["num_digits"] = sum(c.isdigit() for c in url)
    parsed["num_special"] = sum(c in "@?=&%" for c in url)
    parsed["has_ip"] = bool(
        re.search(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", url)
    )
    parsed["has_https"] = url.startswith("https")
    parsed["subdomain_count"] = (
        len(url.split("/")[2].split(".")) - 2
        if "/" in url and len(url.split("/")) > 2
        else 0
    )
    return parsed


# ── Text Utilities ───────────────────────────────────────────────────────────

def clean_html_text(html_content: str) -> str:
    """Extract and clean visible text from raw HTML."""
    try:
        from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
        import warnings
        warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

        soup = BeautifulSoup(html_content, "lxml")

        # Remove non-visible elements
        for tag in soup(["script", "style", "meta", "noscript", "head"]):
            tag.decompose()

        text = soup.get_text(separator=" ")
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text
    except Exception:
        # Fallback: strip HTML tags with regex
        text = re.sub(r"<[^>]+>", " ", html_content)
        return re.sub(r"\s+", " ", text).strip()


# ── HTML Structural Feature Utilities ────────────────────────────────────────

HTML_FEATURES_DIM = 8


def extract_html_features(html_content: str) -> torch.Tensor:
    """
    Extract 8 normalised structural features from raw HTML.
    These capture phishing-specific patterns that DistilBERT (text-only) misses.
    """
    html = str(html_content) if html_content else ""
    html_lower = html.lower()

    form_count     = min(html_lower.count("<form"), 10) / 10.0
    input_count    = min(html_lower.count("<input"), 20) / 20.0
    has_password   = float("password" in html_lower and "<input" in html_lower)
    script_count   = min(html_lower.count("<script"), 20) / 20.0
    iframe_count   = min(html_lower.count("<iframe"), 10) / 10.0
    meta_refresh   = float("http-equiv" in html_lower and "refresh" in html_lower)

    # External link ratio: count href="http" (external) vs total href
    import re as _re
    all_hrefs = _re.findall(r'href\s*=', html_lower)
    ext_hrefs = _re.findall(r'href\s*=\s*["\']https?://', html_lower)
    ext_ratio = len(ext_hrefs) / max(len(all_hrefs), 1)

    # Visible text length (rough proxy — stripped of tags)
    text_only = _re.sub(r'<[^>]+>', ' ', html)
    text_len = min(len(text_only.split()), 2000) / 2000.0

    return torch.tensor(
        [form_count, input_count, has_password, script_count,
         iframe_count, meta_refresh, ext_ratio, text_len],
        dtype=torch.float32,
    )


# ── Image Utilities ──────────────────────────────────────────────────────────

def get_image_transforms(
    image_size: int = 224,
    augment: bool = False,
    augment_config: Optional[dict] = None,
):
    """Return torchvision transforms for screenshot preprocessing."""
    from torchvision import transforms

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet stats
        std=[0.229, 0.224, 0.225],
    )

    if augment and (augment_config or {}).get("enabled", True):
        cfg = augment_config or {}
        crop_scale = tuple(cfg.get("random_resized_crop_scale", [0.85, 1.0]))
        translate = tuple(cfg.get("translate", [0.05, 0.05]))
        zoom_scale = tuple(cfg.get("zoom_scale", [0.9, 1.05]))
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size,
                    scale=crop_scale,
                ),
                transforms.RandomAffine(
                    degrees=float(cfg.get("rotation_degrees", 10)),
                    translate=translate,
                    scale=zoom_scale,
                ),
                transforms.RandomHorizontalFlip(
                    p=float(cfg.get("horizontal_flip_prob", 0.5))
                ),
                transforms.ColorJitter(
                    brightness=float(cfg.get("brightness", 0.2)),
                    contrast=float(cfg.get("contrast", 0.2)),
                    saturation=float(cfg.get("saturation", 0.1)),
                    hue=float(cfg.get("hue", 0.02)),
                ),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )
