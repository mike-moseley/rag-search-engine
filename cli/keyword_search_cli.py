import math
import keyword
import string
import json
import argparse
from nltk.stem import PorterStemmer
from lib.keyword_search import search_command, build_command, tf_command,idf_command, tfidf_command, tokenize_text, InvertedIndex
from lib.search_utils import DEFAULT_SEARCH_LIMIT

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Build Inverted Index")

    tf_parser = subparsers.add_parser("tf", help="Get frequency for a search term")
    tf_parser.add_argument("doc_id", type=int, help="Document Id of movies.json")
    tf_parser.add_argument("term", type=str, help="Term from a search query")

    idf_parser = subparsers.add_parser("idf", help="Get Inverse Document Frequency of term")
    idf_parser.add_argument("term", type=str, help="Term from a search query")

    tfidf_parser = subparsers.add_parser("tfidf", help="Get TFIDF score")
    tfidf_parser.add_argument("doc_id", type=int, help="Document Id of movies.json")
    tfidf_parser.add_argument("term", type=str, help="Term from a search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            # print the search query here
            results = search_command(args.query)
            for i, res in enumerate(results, 1):
                print(f"{i}. {res['title']}")
        case "build":
            print("Building inverted index...")
            build_command()
            print("Inverted index built successfully.")
        case "tf":
            print(f"Getting match frequency of search term: {args.term}")
            print(tf_command(args.doc_id, args.term))
        case "idf":
            idf = idf_command(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tfidf":
            tfidf = tfidf_command(args.doc_id, args.term)
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tfidf:.2f}")

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
