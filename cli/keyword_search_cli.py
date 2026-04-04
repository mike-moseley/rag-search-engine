import math
import keyword
import string
import json
import argparse
from nltk.stem import PorterStemmer
from lib.keyword_search import (
        search_command, build_command, tf_command,
        idf_command, tfidf_command, bm25_idf_command,
        bm25_tf_command, bm25_search_command
)
from lib.keyword_search import tokenize_text, InvertedIndex
from lib.consts import DEFAULT_SEARCH_LIMIT

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

    bm25idf_parser = subparsers.add_parser("bm25idf", help="Get bm25 IDF score")
    bm25idf_parser.add_argument("term", type=str, help="Term from a search query")

    bm25tf_parser = subparsers.add_parser("bm25tf", help="Get bm25 TF score")
    bm25tf_parser.add_argument("doc_id", type=int, help="Document Id of movies.json")
    bm25tf_parser.add_argument("term", type=str, help="Term from a search query")

    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument("--limit", type=int, default=5, required=False, help="Number of results to list" )

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            # print the search query here
            results = search_command(args.query)
            if results is None:
                return
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
        case "bm25idf":
            bm25idf = bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
        case "bm25tf":
            bm25tf = bm25_tf_command(args.doc_id, args.term)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")
        case "bm25search":
            bm25search = bm25_search_command(args.query, args.limit)
            # tup is a 3-tuple (doc_id, movie, bm25score)
            for i, tup in enumerate(bm25search, 1):
                print(f"{i}. ({tup[0]}) {tup[1]} - Score: {tup[2]:.2f}")

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
