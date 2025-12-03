# caption_engine/gemini_writer.py

import os
import google.generativeai as genai

# Configure Gemini with API key from environment (Railway variable)
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable is not set")

genai.configure(api_key=API_KEY)


def rewrite_caption(prompt: str, image_bytes: bytes) -> str:
    """
    Calls Gemini multimodal model with image + text prompt
    and returns a single caption string.
    """

    # Use a multimodal-capable model
    model = genai.GenerativeModel("gemini-2.5-flash")

    # Image part: raw bytes + mime type
    image_part = {
        "mime_type": "image/jpeg",  # works fine for most JPG/PNG uploads
        "data": image_bytes,
    }

    # We send [image, prompt] as content
    response = model.generate_content([image_part, prompt])

    # Extract text safely
    text = getattr(response, "text", "") or ""
    text = text.strip()

    if not text:
        raise RuntimeError("Empty response from Gemini API")

    return text
