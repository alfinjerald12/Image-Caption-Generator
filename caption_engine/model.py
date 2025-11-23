"""
model.py – BLIP generates base caption, Gemini rewrites into styles.
"""

import os
import logging
from PIL import Image

from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

from caption_engine.gemini_writer import rewrite_caption

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_processor = None
_model = None
_device = "cpu"


def _load_model():
    """Lazy-load BLIP model."""
    global _processor, _model, _device

    if _model is None or _processor is None:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading BLIP model on device: {_device}")

        _processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        _model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(_device)

        _model.eval()
        logger.info("BLIP model loaded successfully.")


def _caption_blip(image_path: str, max_length: int = 20) -> str:
    """Get neutral caption from BLIP."""
    image = Image.open(image_path).convert("RGB")
    inputs = _processor(images=image, return_tensors="pt").to(_device)

    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_length=max_length,
            num_beams=5,
        )

    text = _processor.decode(output_ids[0], skip_special_tokens=True).strip()

    if text and not text.endswith("."):
        text = text[0].upper() + text[1:] + "."
    return text


def generate_caption(image_path: str, style: str = "realistic") -> str:
    """
    Full pipeline: BLIP → Gemini style rewrite.
    If Gemini fails, we fall back to BLIP caption.
    """

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # 1) BLIP caption
    _load_model()
    base_caption = _caption_blip(image_path)
    print(">> BLIP base caption:", base_caption)

    # 2) Gemini rewrite
    try:
        style = (style or "realistic").lower()
        if style in ["realistic", "funny", "sad"]:
            styled = rewrite_caption(base_caption, style)
            if styled:
                return styled
        # if style unknown or rewrite empty, use BLIP
        return base_caption
    except Exception as e:
        print(">> Gemini rewrite failed, using BLIP caption only:", e)
        return base_caption


# Optional CLI test
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Caption generator with BLIP + Gemini")
    parser.add_argument("image", help="Path to the image file")
    parser.add_argument(
        "--style",
        choices=["realistic", "funny", "sad"],
        default="realistic",
        help="Caption style",
    )
    args = parser.parse_args()

    print("Style:", args.style)
    print("Caption:", generate_caption(args.image, style=args.style))
