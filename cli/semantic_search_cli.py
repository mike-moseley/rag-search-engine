#!/usr/bin/env python3

import argparse
from lib.semantic_search import embed_query_text, verify_model, embed_text, verify_embeddings, search_command, chunk_command, chunk_command_text, embed_chunks_command, search_chunked_command

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="available commands")

    subparsers.add_parser("verify", help="Print the current model and max sequence length of the model")

    embed_text_parser = subparsers.add_parser("embed_text", help="Embed text into model")
    embed_text_parser.add_argument("text", help="Text to embed")

    subparsers.add_parser("verify_embeddings", help="Verify current embeddings")

    embed_query_parser = subparsers.add_parser("embedquery", help="Embed a search query")
    embed_query_parser.add_argument("query", type=str, help="Query to embed")

    search_parser = subparsers.add_parser("search", help="Returns <limit> results matching <query>")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("--limit", type=int, default=5, required=False, help="Number of results to display")

    chunk_parser = subparsers.add_parser("chunk", help="Chunk data")
    chunk_parser.add_argument("text", type=str, help="Data to chunk as str")
    chunk_parser.add_argument("--chunk-size", type=int, default=200, required=False, help="Size in words of each chunk")
    chunk_parser.add_argument("--overlap", type=int, default=0, required=False, help="Number of words to overlap" )

    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="Chunks data such that chunks retain meaning")
    semantic_chunk_parser.add_argument("text", type=str, help="Data to chunk as str")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, default=4, required=False, help="Max size of each chunk")
    semantic_chunk_parser.add_argument("--overlap", type=int, default=0, required=False, help="Number of words to overlap")

    embed_chunks_parser = subparsers.add_parser("embed_chunks", help="Embed chunks")

    search_chunked_parser = subparsers.add_parser("search_chunked", help="Search on chunked data")
    search_chunked_parser.add_argument("query", type=str, help="Query to search for")
    search_chunked_parser.add_argument("--limit", type=int, default=5, required=False)

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embedquery":
            embed_query_text(args.query)
        case "search":
            search_command(args.query, args.limit)
        case "chunk":
            chunks = chunk_command(args.text, args.chunk_size, args.overlap, False)
            chunk_command_text(args.text, chunks, False)
        case "semantic_chunk":
            chunks = chunk_command(args.text, args.max_chunk_size, args.overlap, True)
            chunk_command_text(args.text, chunks, True)
        case "embed_chunks":
            embed_chunks_command()
        case "search_chunked":
            search_chunked_command(args.query, args.limit)

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
