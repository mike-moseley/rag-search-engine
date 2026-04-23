import argparse
import json
import logging
from lib.consts import GOLDEN_DATA_PATH
from lib.hybrid_search import rrf_search_get
from lib.search_utils import load_movies

def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    with open(GOLDEN_DATA_PATH) as f:
        data = json.load(f)["test_cases"]

    print(f"k = {limit}")
    for test in data:
        scores = rrf_search_get(test["query"], 60, limit, None, "cross_encoder", None)

        retrieved = []
        relevant_num = 0
        for score in scores:
            if score["title"] in test["relevant_docs"]:
                relevant_num += 1
            retrieved.append(score["title"])

        recall = relevant_num/len(test['relevant_docs'])
        precision = relevant_num / limit
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        print(f"- Query: {test['query']}")
        print(f"  - Precision@{limit}: {precision:.4f}")
        print(f"  - Recall@{limit}: {recall:.4f}")
        print(f"  - F1 Score: {f1:.4f}")
        print(f"  - Retrieved: {', '.join(retrieved)}")
        print(f"  - Relevant: {', '.join(test['relevant_docs'])}")

if __name__ == "__main__":
    main()
