import google.generativeai as genai
import os

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def rewrite_caption(prompt, image_base64):
    model = genai.GenerativeModel("models/gemini-2.5-flash-image")

    response = model.generate_content(
        [
            {"text": prompt},
            {
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": image_base64
                }
            }
        ]
    )

    return response.text.strip()
