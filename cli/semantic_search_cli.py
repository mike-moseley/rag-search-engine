#!/usr/bin/env python3

import argparse
from lib.semantic_search import verify_model, embed_text, verify_embeddings

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="available commands")

    verify_parser = subparsers.add_parser("verify", help="Print the current model and max sequence length of the model")

    embed_text_parser = subparsers.add_parser("embed_text", help="Embed text into model")
    embed_text_parser.add_argument("text", help="Text to embed")

    verify_embeddings_parser = subparsers.add_parser("verify_embeddings", help="Verify current embeddings")
    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
