import os

from .search_utils import load_movies
from .hybrid_search import rrf_search_get, HybridSearch
from .ai_prompts import RAG, RAG_SUMMARIZE, RAG_CITATION, RAG_QUESTION
from google import genai
from dotenv import load_dotenv

def setup_rag(query: str, RAG_PROMPT):
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")
    movies = load_movies()
    hs = HybridSearch(movies)
    results = hs.rrf_search(query, 60, 5)
    client = genai.Client(api_key=api_key)
    resp = client.models.generate_content(
        model="gemma-3-27b-it",
        contents=RAG_PROMPT(query, results)
    )
    return (results[:5], resp)

def rag_command(query: str):
    # load_dotenv()
    # api_key = os.environ.get("GEMINI_API_KEY")
    # if not api_key:
    #     raise RuntimeError("GEMINI_API_KEY environment variable not set")
    #
    # client = genai.Client(api_key=api_key)
    # docs = rrf_search_get(query, 60, 5, None, None, None)
    # resp = client.models.generate_content(
    #     model="gemma-3-27b-it",
    #     contents=RAG(query, docs)
    # )
    movies, resp = setup_rag(query, RAG)

    print("Search Results:")
    for m in movies:
        print(f"- {m['title']}")

    print()
    print("RAG Response:")
    print(f"{resp.text}")

def summarize_command(query: str):
    # load_dotenv()
    # api_key = os.environ.get("GEMINI_API_KEY")
    # if not api_key:
    #     raise RuntimeError("GEMINI_API_KEY environment variable not set")
    # movies = load_movies()
    # hs = HybridSearch(movies)
    # results = hs.rrf_search(query, 60, 5)
    # client = genai.Client(api_key=api_key)
    # resp = client.models.generate_content(
    #     model="gemma-3-27b-it",
    #     contents=RAG_SUMMARIZE(query, results)
    # )
    movies, resp = setup_rag(query, RAG_SUMMARIZE)

    print("Search Results:")
    for m in movies:
        print(f"- {m['title']}")

    print()
    print("LLM Summary:")
    print(f"{resp.text}")

def citations_command(query: str):
    # load_dotenv()
    # api_key = os.environ.get("GEMINI_API_KEY")
    # if not api_key:
    #     raise RuntimeError("GEMINI_API_KEY environment variable not set")
    # movies = load_movies()
    # hs = HybridSearch(movies)
    # results = hs.rrf_search(query, 60, 5)
    # client = genai.Client(api_key=api_key)
    # resp = client.models.generate_content(
    #     model="gemma-3-27b-it",
    #     contents=RAG_CITATION(query, results)
    # )
    movies, resp = setup_rag(query, RAG_CITATION)

    print("Search Results:")
    for m in movies:
        print(f"- {m['title']}")

    print()
    print("LLM Answer:")
    print(f"{resp.text}")

def question_command(query: str):
    movies, resp = setup_rag(query, RAG_QUESTION)

    print("Search Results:")
    for m in movies:
        print(f"- {m['title']}")

    print()
    print("Answer:")
    print(f"{resp.text}")
