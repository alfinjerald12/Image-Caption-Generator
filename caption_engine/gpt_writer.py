import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def rewrite_caption(base_caption: str, style: str) -> str:
    """
    Rewrite BLIP caption using GPT for a creative style.
    If GPT fails, caller should fall back to base_caption.
    """
    prompt = f"""
    Rewrite the following short image caption in a **{style}** style.
    Caption: "{base_caption}"

    Rules:
    - Keep it concise (max 1 sentence).
    - Do NOT mention that this is a caption.
    - Do NOT describe the rewriting process.
    - Make it natural, expressive, and human-like.
    - Do NOT add emojis.
    """

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=1.0,
    )

    text = response.choices[0].message.content.strip()
    if not text.endswith("."):
        text += "."
    return text
