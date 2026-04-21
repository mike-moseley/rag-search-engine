import argparse
from lib.rag import rag_command, summarize_command, citations_command, question_command

def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser("summarize", help="Provide summarized RAG")
    summarize_parser.add_argument("query", type=str, help="Search query")

    citations_parser = subparsers.add_parser("citations", help="Provide summary with citations")
    citations_parser.add_argument("query", type=str, help="Search query")

    question_parser = subparsers.add_parser("question", help="Provide answer to question about the movies")
    question_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            rag_command(query)

        case "summarize":
            summarize_command(args.query)

        case "citations":
            citations_command(args.query)

        case "question":
            question_command(args.query)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
