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


# ── Image Utilities ──────────────────────────────────────────────────────────

def get_image_transforms(image_size: int = 224, augment: bool = False):
    """Return torchvision transforms for screenshot preprocessing."""
    from torchvision import transforms

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet stats
        std=[0.229, 0.224, 0.225],
    )

    if augment:
        return transforms.Compose(
            [
                transforms.Resize((image_size + 32, image_size + 32)),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
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
