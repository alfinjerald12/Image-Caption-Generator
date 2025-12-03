# caption_engine/gemini_writer.py

import os
import google.generativeai as genai

# Try both common env variable names
API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

if not API_KEY:
    # This will show clearly in Railway logs if key is missing
    raise RuntimeError(
        "No Gemini API key found. "
        "Set GOOGLE_API_KEY (recommended) or GEMINI_API_KEY in your environment."
    )

# Configure Gemini client
genai.configure(api_key=API_KEY)


def rewrite_caption(prompt: str, image_bytes: bytes) -> str:
    """
    Calls Gemini multimodal model (image + text) and returns a single caption.
    """
    # Use a multimodal-capable model
    model = genai.GenerativeModel("gemini-2.5-flash")

    image_part = {
        "mime_type": "image/jpeg",  # works for most uploads
        "data": image_bytes,
    }

    response = model.generate_content([image_part, prompt])

    text = getattr(response, "text", "") or ""
    text = text.strip()

    if not text:
        raise RuntimeError("Empty response from Gemini API")

    return text
