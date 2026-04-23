import argparse
import mimetypes
import os

from google import genai
from dotenv import load_dotenv
from lib.ai_prompts import RAG_IMAGE

def main():
    parser = argparse.ArgumentParser(description="Describe Image Search")
    parser.add_argument("--image", help="Path to image")
    parser.add_argument("--query", help="Text query")

    args = parser.parse_args()

    mime, _ = mimetypes.guess_type(args.image)

    mime = mime or "image/jpeg"

    with open(args.image, "rb") as f:
        img = f.read()

    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")
    client = genai.Client(api_key=api_key)
    parts = [
        RAG_IMAGE(),
        genai.types.Part.from_bytes(data=img, mime_type=mime),
        args.query.strip(),
    ]
    resp = client.models.generate_content(
        model="gemma-3-27b-it",
        contents=(parts)
    )
    assert resp.text is not None
    print(f"Rewritten query: {resp.text.strip()}")
    if resp.usage_metadata is not None:
        print(f"Total tokens:    {resp.usage_metadata.total_token_count}")

if __name__ == "__main__":
    main()
