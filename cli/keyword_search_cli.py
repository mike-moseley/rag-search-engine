import keyword
import string
import json
import argparse
import lib.keyword_search
import lib.search_utils

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Build Inverted Index")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            # print the search query here
            results = lib.keyword_search.search_command(args.query)
            for i, res in enumerate(results, 1):
                print(f"{i}. {res['title']}")
        case "build":
            print("Building inverted index...")
            lib.keyword_search.build_command()
            print("Inverted index built successfully.")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
