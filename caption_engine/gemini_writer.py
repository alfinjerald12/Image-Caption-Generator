import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load env vars
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY not found in .env file")

genai.configure(api_key=api_key)

# ✔ Best supported text model for your account
_model = genai.GenerativeModel("models/gemini-flash-latest")


def rewrite_caption(base_caption: str, style: str) -> str:
    """
    Use Gemini to rewrite BLIP's caption in a chosen style.
    Styles: realistic, funny, sad
    """

    style = (style or "realistic").lower()

    prompt = f"""
Rewrite the following short image caption into a {style} style.

Caption: "{base_caption}"

Rules:
- Output exactly ONE sentence.
- Do NOT mention the rewriting process.
- Do NOT mention it’s a caption.
- Make it expressive and humanlike.
- Avoid emojis.
- Keep it under 20 words.
"""

    response = _model.generate_content(prompt)
    text = (response.text or "").strip()

    if text and not text.endswith("."):
        text += "."
    return text
