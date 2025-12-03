# caption_engine/model.py

from caption_engine.gemini_writer import rewrite_caption


def generate_caption(image_path: str, style: str = "realistic") -> str:
    """
    Generate a caption for an image using only Gemini (no PyTorch, no BLIP).
    The image is read as raw bytes and passed directly to Gemini.
    """

    # Read image as bytes
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    # Style-specific instruction
    style_instruction = {
        "realistic": "Describe the image in a clear, detailed and natural way.",
        "funny": "Describe the image with a playful, witty and humorous tone.",
        "sad": (
            "Describe the image with a melancholic, reflective tone, "
            "without using the words 'sad', 'depressed' or 'emotional'."
        ),
    }.get(style, "Describe the image in a natural and clear way.")

    # Prompt sent to Gemini
    prompt = f"""
You are an image captioning assistant.

Image style requested: {style}

Instructions:
- Look at the provided image and generate ONE single-line caption.
- {style_instruction}
- The caption must be specific to the actual scene, not generic.
- Do not mention that you are an AI or that you are looking at an image.
- Do not mention the style name (funny/realistic/sad) in the caption.
"""

    return rewrite_caption(prompt, image_bytes)
