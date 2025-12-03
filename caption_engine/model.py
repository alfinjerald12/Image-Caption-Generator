import base64
from caption_engine.gemini_writer import rewrite_caption

def generate_caption(image_path, style="realistic"):
    """
    Generates an image caption using ONLY Gemini (no torch, no BLIP).
    Image is converted to base64 and sent to Gemini for understanding.
    """

    with open(image_path, "rb") as img:
        image_bytes = img.read()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    prompt = f"""
Analyze this image carefully and generate a single {style} caption.

Rules:
- If style is realistic → describe clearly and naturally.
- If style is funny → add humor based on what you see.
- If style is sad → generate a deep, emotional, melancholic tone.
- The caption must change based on the image.
- Do not reuse generic phrases.
"""

    return rewrite_caption(prompt, image_base64)
