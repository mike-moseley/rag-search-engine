import json
from time import sleep
import os
import logging

from sentence_transformers import CrossEncoder
from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from dotenv import load_dotenv
from google import genai
from .consts import INDEX_PICKLE_PATH
from .ai_prompts import RRF_ENHANCE_SPELL, RRF_ENHANCE_REWRITE, RRF_ENHANCE_EXPAND, RRF_RERANK_INDIVIDUAL, RRF_RERANK_BATCH, RRF_EVALUATE
from .search_utils import load_movies

logger = logging.getLogger(__name__)
logging.basicConfig(filename='rrf-search.log', encoding='utf-8', level = logging.DEBUG)

class HybridSearch:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(INDEX_PICKLE_PATH):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int) -> list[dict]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[dict]:
        bm25_res = self._bm25_search(query, limit * 500)
        bm25_scores = {result["id"]: result["score"] for result in bm25_res}

        sem_res = self.semantic_search.search_chunks(query, limit * 500)
        sem_scores = {result["id"]: result["score"] for result in sem_res}

        combined: dict[int, dict] = {}
        for i, doc_id in enumerate(bm25_scores, 1):
            combined[doc_id] = {"bm25_rank": i,}
        for i, doc_id in enumerate(sem_scores, 1):
            if doc_id in combined:
                combined[doc_id]["sem_rank"] = i
            else:
                combined[doc_id] = {"sem_rank": i}
        scores_list = []
        for doc_id, ranks in combined.items():
            bm25_rank = ranks.get("bm25_rank", 0)
            sem_rank = ranks.get("sem_rank", 0)
            score = 0.0
            if bm25_rank:
                score += rrf_score(bm25_rank, k)
            if sem_rank:
                score += rrf_score(sem_rank, k)
            scores_list.append({
                "movie_id": doc_id,
                "title": self.idx.docmap[doc_id]["title"],
                "score": score,
                "bm25_rank": bm25_rank,
                "sem_rank": sem_rank,
                "document": self.idx.docmap[doc_id]["description"]
            })
        sorted_scores = sorted(scores_list, key=lambda x: x["score"], reverse=True)[:limit]
        logger.info(f"Sorted Scores: {sorted_scores}")
        return sorted_scores

    def weighted_search(self, query: str, alpha: float, limit: int) -> list[dict]:
        bm25_res = self._bm25_search(query, limit * 500)
        bm25_scores = {result["id"]: result["score"] for result in bm25_res}
        norm_bm25 = normalize_command(bm25_scores)

        sem_res = self.semantic_search.search_chunks(query, limit * 500)
        sem_scores = {result["id"]: result["score"] for result in sem_res}
        norm_sem = normalize_command(sem_scores)

        combined: dict[int, dict] = {}
        for doc_id, bm25_score in norm_bm25.items():
            combined[doc_id] = {"bm25_score": bm25_score, "semantic_score": 0.0}
        for doc_id, sem_score in norm_sem.items():
            if doc_id in combined:
                combined[doc_id]["semantic_score"] = sem_score
            else:
                combined[doc_id] = {"bm25_score": 0.0, "semantic_score": sem_score}

        scores_list = []
        for doc_id, score in combined.items():
            scores_list.append({
                "movie_id": doc_id,
                "title": self.idx.docmap[doc_id]["title"],
                "bm25_score": score["bm25_score"],
                "semantic_score": score["semantic_score"],
                "hybrid_score": hybrid_score(score["bm25_score"], score["semantic_score"], alpha)
            })
        sorted_scores = sorted(scores_list, key=lambda x: x["hybrid_score"], reverse=True)[:limit]
        logger.info(f"Sorted Scores: {sorted_scores}")
        return sorted_scores


def normalize_command(scores: dict[int, float]) -> dict[int, float]:
    if not scores:
        return {}
    min_score = min(scores.values())
    max_score = max(scores.values())
    if min_score == max_score:
        return {k: 1.0 for k in scores}
    return {k: (v - min_score) / (max_score - min_score) for k, v in scores.items()}

def hybrid_score(bm25_score: float, semantic_score: float, alpha: float = 0.5) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score

def weighted_search_command(query: str, alpha: float, limit: int) -> None:
    movies = load_movies()
    hs = HybridSearch(movies)
    scores = hs.weighted_search(query, alpha, limit)
    for i, score in enumerate(scores, 1):
        print(f"{i}. {score['title']}")
        print(f"  Hybrid Score: {score['hybrid_score']:.3f}")
        print(f"  BM25: {score['bm25_score']:.3f}, Semantic: {score['semantic_score']:.3f}")

def rrf_search_get(query: str, k:int, limit:int, enhance: str | None, rerank: str | None, client: genai.Client | None):
    logger.info(f"Original Query: {query}")
    movies = load_movies()
    hs = HybridSearch(movies)
    if rerank:
        original_limit = limit
        limit = limit * 5

    enhanced_query = None
    if client:
        match(enhance):
            case "spell":
                resp = client.models.generate_content(
                    model="gemma-3-27b-it",
                    contents=RRF_ENHANCE_SPELL(query)
                )
                enhanced_query = resp.text
            case "rewrite":
                resp = client.models.generate_content(
                    model="gemma-3-27b-it",
                    contents=RRF_ENHANCE_REWRITE(query)
                )
                enhanced_query = resp.text
            case "expand":
                resp = client.models.generate_content(
                    model="gemma-3-27b-it",
                    contents=RRF_ENHANCE_EXPAND(query)
                )
                if resp.text is not None:
                    enhanced_query = resp.text + " " + query

    if enhanced_query:
        logger.info(f"Enhanced Query: {enhanced_query}")

    if enhance:
        if resp is None or enhanced_query is None:
            raise RuntimeError("Failed to get response from AI with enhanced query")
        scores = hs.rrf_search(enhanced_query, k, limit)
        print(f"Enhanced query({enhance}): '{query}' -> '{enhanced_query}'")
    else:
        scores = hs.rrf_search(query, k, limit)

    logger.info(f"Search Results: {scores}")

    if enhanced_query:
        query = enhanced_query
    match(rerank):
        case "individual":
            if client:
                scores = individual_rerank(query, client, scores, original_limit)
        case "batch":
            if client:
                scores = batch_rerank(query, client, scores, original_limit)
        case "cross_encoder":
            scores = cross_encoder_rerank(query, scores, original_limit)

    logger.info(f"Reranked Results: {scores[:limit]}")
    return scores[:limit]

def rrf_score(rank, k=60):
    return 1 / (k + rank)

def f_string_rrf(scores: list, rerank: str | None):
    formatted_results = []
    for i, score in enumerate(scores, 1):
        formatted_results.append(f'{i}. {score["title"]}')
        if rerank:
            if rerank == "cross_encoder":
                formatted_results.append(f"  Cross Encoder Score: {score["rerank-score"]}")
            else:
                formatted_results.append(f"  Re-rank Score: {score["rerank-score"]}")
        formatted_results.append(f"  RRF Score: {score["score"]}")
        formatted_results.append(f"  BM25 Rank: {score["bm25_rank"]}, Semantic Rank: {score["sem_rank"]}")
        formatted_results.append(f"  {score["document"]}")
    return formatted_results

def individual_rerank(query: str, client: genai.Client, movies: list[dict], limit: int):
    results = []
    for movie in movies:
        resp = client.models.generate_content(
            model="gemma-3-27b-it",
            # resp.text should contain only a score as a str
            contents=RRF_RERANK_INDIVIDUAL(query, movie)
        )
        if not resp or resp.text is None:
            raise RuntimeError("Invalid response from AI in individual_rerank()")
        movie["rerank-score"] = int(resp.text.strip())
        results.append(movie)
        sleep(3)
    return sorted(results, key=lambda x: x["score"], reverse=True)[:limit]

def batch_rerank(query: str, client: genai.Client, movies: list[dict], limit: int):
    respJson = client.models.generate_content(
        model="gemma-3-27b-it",
        contents=RRF_RERANK_BATCH(query, movies)
    )
    if respJson.text is None:
        raise RuntimeError("AI response text field is empty")
    resp = json.loads(respJson.text)
    for i, movie_id in enumerate(resp, 1):
        for movie in movies:
            if movie["movie_id"] == movie_id:
                movie["rerank-score"] = i
                break
    return sorted(movies, key=lambda x: x["rerank-score"])[:limit]

def cross_encoder_rerank(query: str, movies: list[dict], limit: int):
    pairs = []
    for movie in movies:
        pairs.append([query, f"{movie.get('title', '')} - {movie.get('document', '')}"])

    cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")

    scores = cross_encoder.predict(pairs)
    for score, movie in zip(scores, movies):
        movie["rerank-score"] = score

    return sorted(movies, key=lambda x: x["rerank-score"], reverse=True)[:limit]

def evaluate_rrf(query: str,scores: list, scores_f_strings: list, client: genai.Client):
    resp = client.models.generate_content(
        model="gemma-3-27b-it",
        contents=RRF_EVALUATE(query, scores_f_strings)
    )


    if resp.text is None:
        raise RuntimeError("Empty response from AI in evaluate_rrf")

    for i, (score, eval) in enumerate(zip(scores,json.loads(resp.text)), 1):
        print(f"{i}. {score["title"]}: {eval}/3")
    
def print_rrf(query: str, k:int, limit:int, enhance: str | None, rerank: str | None, evaluate: bool):
    client = None
    if enhance or rerank or evaluate:
        load_dotenv()
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable not set")

        client = genai.Client(api_key=api_key)
    scores = rrf_search_get(query, k, limit, enhance, rerank, client)
    formatted = f_string_rrf(scores, rerank)
    print('\n'.join(formatted))
    print()
    if evaluate and client:
        evaluate_rrf(query, scores, formatted, client)
