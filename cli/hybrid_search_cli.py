import argparse
from lib.hybrid_search import normalize_command, weighted_search_command, rrf_search_get, f_string_rrf, evaluate_rrf, print_rrf

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="Return normalized list")
    normalize_parser.add_argument("list",type=float, nargs="+")

    weighted_search = subparsers.add_parser("weighted-search", help="Search using weighted results between bm25 and semantic search")
    weighted_search.add_argument("query", type=str)
    weighted_search.add_argument("--alpha", type=float, default=0.5, help="Weight to use")
    weighted_search.add_argument("--limit", type=int, default=5, help="Number of results to return")

    rrf_search = subparsers.add_parser("rrf-search", help="rrf-search")
    rrf_search.add_argument("query", type=str)
    rrf_search.add_argument("-k", type=int, default=60, required=False)
    rrf_search.add_argument("--limit", type=int, default=5, required=False)
    rrf_search.add_argument("--enhance", type=str, choices=["spell", "rewrite", "expand"], help="Query enchancement method")
    rrf_search.add_argument("--rerank-method", type=str, choices=["individual", "batch", "cross_encoder"], help="Results reranking method")
    rrf_search.add_argument("--evaluate", action="store_true")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalize_command(args.list)
        case "weighted-search":
            weighted_search_command(args.query, args.alpha, args.limit)
        case "rrf-search":
            print_rrf(args.query, args.k, args.limit, args.enhance, args.rerank_method, args.evaluate)

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
