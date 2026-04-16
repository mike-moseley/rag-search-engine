import json
import re
import os
from sentence_transformers import SentenceTransformer
from lib.consts import EMBEDDING_PATH, CHUNK_EMBEDDING_PATH, CHUNK_METADATA_PATH, SCORE_PRECISION
from lib.search_utils import load_movies
import numpy as np

class SemanticSearch:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text: str):
        if text.strip() == "":
            raise ValueError("Text cannot be empty or just whitespace")

        return self.model.encode([text])[0]

    def build_embeddings(self, documents: list[dict[str, str]]):
        self.populate_doc_doc_map(documents)

        movie_list = []
        for doc in documents:
            movie_list.append(f"{doc['title']}: {doc['description']}")

        embeddings = self.model.encode(movie_list, show_progress_bar=True)
        self.embeddings = embeddings
        np.save(EMBEDDING_PATH, embeddings)

        return self.embeddings

    def load_or_create_embeddings(self, documents: list[dict[str,str]]):
        if os.path.exists(EMBEDDING_PATH):
            self.populate_doc_doc_map(documents)
            embeddings = np.load(EMBEDDING_PATH)
            self.embeddings = embeddings
            if len(self.embeddings) == len(documents):
                return self.embeddings
        return self.build_embeddings(documents)

    def populate_doc_doc_map(self, documents: list[dict[str,str]]):
        self.documents = documents
        for doc in documents:
            self.document_map[doc['id']] = doc

    def search(self, query: str, limit: int) -> list[dict] | None:
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        if self.documents is None:
            raise ValueError("No documents loaded. Call `populate_dod_doc_map` first")
        embedded_query = self.generate_embedding(query)
        scores: list[tuple] = []
        for i,e in enumerate(self.embeddings):
            scores.append((cosine_similarity(e,embedded_query), i))

        scores.sort(key=lambda x:x[0], reverse=True)
        embedded_dicts: list[dict[str, float | str]] = []
        for score, i in scores[:limit]:
            doc = self.documents[i]
            embedded_dicts.append({
                "score": score,
                "title": doc["title"],
                "description": doc["description"],
            })

        return embedded_dicts

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None
    def build_chunk_embeddings(self, documents: list[dict[str,str]]):
        self.populate_doc_doc_map(documents)

        chunk_list: list[str] = []
        metadata_list: list[dict] = []

        for i,doc in enumerate(documents):
            if doc["description"].strip() == "":
                continue
            chunks = chunk_command(doc["description"], 4, 1, True)
            chunk_list.extend(chunks)
            for j, chunk in enumerate(chunks):
                metadata_list.append({"movie_idx": i, "chunk_idx": j, "total_chunks": len(chunks)})

        chunk_embeddings = self.model.encode(chunk_list, show_progress_bar=True)
        self.chunk_embeddings = chunk_embeddings
        self.chunk_metadata = metadata_list
        np.save(CHUNK_EMBEDDING_PATH, chunk_embeddings)
        with open(CHUNK_METADATA_PATH, "w") as f:
            json.dump({"chunks": metadata_list, "total_chunks": len(chunk_list)}, f, indent=2)
        return chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict[str,str]]):
        self.populate_doc_doc_map(documents)
        if os.path.exists(CHUNK_EMBEDDING_PATH) and os.path.exists(CHUNK_METADATA_PATH):
            chunk_embeddings = np.load(CHUNK_EMBEDDING_PATH)
            self.chunk_embeddings = chunk_embeddings
            with open(CHUNK_METADATA_PATH) as f:
                chunk_metadata = json.load(f)
                self.chunk_metadata = chunk_metadata["chunks"]
            return self.chunk_embeddings
        return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10):
        embedding = self.generate_embedding(query)
        chunk_scores = []
        if self.chunk_embeddings is None:
            raise ValueError("No chunk embeddings loaded")
        if self.chunk_metadata is None:
            raise ValueError("No chunk metadata loaded")
        for i, chunk in enumerate(self.chunk_embeddings):
            score = cosine_similarity(embedding, chunk)
            chunk_scores.append({
                "chunk_idx": i,
                "movie_idx": self.chunk_metadata[i]["movie_idx"],
                "score": score
            })

        movie_scores = {}
        for chunk in chunk_scores:
            movie_idx = chunk["movie_idx"]
            if movie_idx not in movie_scores or chunk["score"] > movie_scores[movie_idx]:
                movie_scores[movie_idx] = chunk["score"]
        sorted_scores = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)
        if sorted_scores is None:
            raise ValueError("No scores after sorting")
        final_list = []
        for score in sorted_scores:
            if self.documents is None:
                raise ValueError("Documents is empty")
            movie = self.documents[score[0]]
            final_list.append({
                "id": movie["id"],
                "title": movie["title"],
                "document": movie["description"][:100],
                "score": round(score[1], SCORE_PRECISION),
                "metadata": movie.get("metadata") or {}
            })
        return final_list[:limit]


def verify_model():
    semantic_search = SemanticSearch()
    print(f"Model loaded: {semantic_search.model}")
    print(f"Max sequence length: {semantic_search.model.get_max_seq_length()}")

def embed_text(text: str):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():
    semantic_search = SemanticSearch()
    movies = load_movies()
    if movies is None:
        raise ValueError("Failed to load movies")
    embedding = semantic_search.load_or_create_embeddings(movies)
    documents = semantic_search.documents
    if documents is None:
        raise ValueError("Failed to get/create embedding")
    print(f"Number of docs: {len(documents)}")
    print(f"Embeddings shape: {embedding.shape[0]} vectors in {embedding.shape[1]} dimensions")

def embed_query_text(query: str):
    semantic_search = SemanticSearch()
    print(f"Query: {query}")
    embedding = semantic_search.generate_embedding(query)
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape[0]}")

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def search_command(query:str, limit:int):
    semantic_search = SemanticSearch()
    movies = load_movies()
    semantic_search.load_or_create_embeddings(movies)
    results = semantic_search.search(query, limit)
    if results == None:
        raise ValueError("Searching failed, results empty")
    for i,r in enumerate(results, 1):
        print(f'{i}. {r["title"]} ({r["score"]})\n\t{r["description"][:80]}...')

def chunk_command(query: str, size: int, overlap: int, semantic: bool):
    query = query.strip()
    if query == "":
        return []
    if semantic == True:
        split = re.split(r"(?<=[.!?])\s+", query)
    else:
        split = query.split()
    if len(split) == 1 and not split[0].endswith(("!","?",".")):
        return [query]
    chunks: list[str] = []
    step = size - overlap
    for i in range(0, len(split), step):
        chunk = " ".join(split[i:i+size])
        chunk = chunk.strip()
        if chunk == "":
            continue
        if len(split[i:i+size]) > overlap:
            chunks.append(chunk)
    return chunks

def chunk_command_text(query: str, chunks: list[str], semantic: bool):
    if semantic == True:
        print(f"Semantically chunking {len(query)} characters:")
    else:
        print(f"Chunking {len(query)} characters:")
    for i,chunk in enumerate(chunks, 1):
        print(f"{i}. {chunk}")

def embed_chunks_command():
    chunked_semantic_search = ChunkedSemanticSearch()
    movies = load_movies()
    embeddings = chunked_semantic_search.load_or_create_chunk_embeddings(movies)
    print(f"Generated {len(embeddings)} chunked embeddings")

def search_chunked_command(query: str, limit: int):
    chunked_semantic_search = ChunkedSemanticSearch()
    movies = load_movies()
    embeddings = chunked_semantic_search.load_or_create_chunk_embeddings(movies)
    search = chunked_semantic_search.search_chunks(query, limit)
    for i,s in enumerate(search,1):
        print(f"\n{i}. {s["title"]} (score: {s["score"]:.4f})")
        print(f"    {s["document"]}...")
