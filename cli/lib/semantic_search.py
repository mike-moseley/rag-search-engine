import os
from sentence_transformers import SentenceTransformer
from lib.consts import EMBEDDING_PATH
from lib.search_utils import load_movies
import numpy as np

class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
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

def chunk_command(query: str, size: int, overlap: int = 0):
    split = query.split()
    chunks: list[str] = []
    step = size - overlap
    for i in range(0, len(split), step):
        chunk = " ".join(split[i:i+size])
        if len(split[i:i+size]) > overlap:
            chunks.append(chunk)

    print(f"Chunking {len(query)} characters:")
    for i,chunk in enumerate(chunks, 1):
        print(f"{i}. {chunk}")
