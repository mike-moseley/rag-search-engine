import argparse
from lib.multimodal_search import verify_image_embedding, image_search_command

def main():
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser("verify_image_embedding")
    verify_parser.add_argument("image_path", type=str)

    image_search_parser = subparsers.add_parser("image_search")
    image_search_parser.add_argument("image_path", type=str)

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.image_path)
        case "image_search":
            results = image_search_command(args.image_path)[:10]
            for i,r in enumerate(results,1):
                print(f"{i}. {r['title']} (similarity: {r['score']:.3f})")
                print(f"   {r['description'][:100]}")


        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
