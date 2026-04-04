import math
from collections import Counter
import pickle
import os
import string
from nltk.stem import PorterStemmer
from lib.search_utils import load_movies, load_stopwords
from lib.consts import *

class InvertedIndex:
    def __init__(self) -> None:
        self.index: dict[str, set[int]] = {}
        self.docmap: dict[int, dict] = {}
        self.term_frequencies: dict[int, Counter] = {}
        self.doc_lengths: dict[int, int] = {}

    def __add_documents(self, doc_id: int, text: str) -> None:
        tokens = tokenize_text(text)
        self.doc_lengths[doc_id] = len(tokens)
        if doc_id not in self.term_frequencies:
            self.term_frequencies[doc_id] = Counter()
        for token in set(tokens):
            if token not in self.index:
                self.index[token] = {doc_id}
            else:
                self.index[token].add(doc_id)

        for token in tokens:
            self.term_frequencies[doc_id][token] += 1

    def __get_avg_doc_length(self) -> float:
        sum = 0
        for doc in self.docmap:
            sum += self.doc_lengths[doc]

        return sum / len(self.docmap)

    def get_documents(self, term: str) -> list[int]:
        term_tokens = tokenize_text(term)
        if len(term_tokens) != 1:
            raise ValueError(f"Expected 1 result, got {len(term_tokens)}")
        docs = self.index.get(term_tokens[0], set())
        return sorted(list(docs))

    def get_tf(self, doc_id: int, term: str) -> int:
        term_tokens = tokenize_text(term)
        if len(term_tokens) != 1:
            raise ValueError(f"Expected 1 result, got {len(term_tokens)}")
        return self.term_frequencies[doc_id][term_tokens[0]]

    def get_idf(self, term: str) -> float:
        term_tokens = tokenize_text(term)
        if len(term_tokens) != 1:
            raise ValueError(f"Expected 1 result, got {len(term_tokens)}")
        total_doc_count = len(self.docmap)
        term_match_doc_count = len(self.index.get(term_tokens[0], set()))
        return math.log((total_doc_count + 1) / (term_match_doc_count + 1))

    def get_bm25_idf(self, term: str) -> float:
        term_tokens = tokenize_text(term)
        if len(term_tokens) != 1:
            raise ValueError(f"Expected 1 result, got {len(term_tokens)}")
        total_doc_count = len(self.docmap)
        term_match_doc_count = len(self.index.get(term_tokens[0], set()))

        return math.log((total_doc_count - term_match_doc_count + 0.5) / (term_match_doc_count + 0.5) + 1)

    def get_bm25_tf(self, doc_id: int, term: str, k1:float=BM25_K1, b:float=BM25_B) -> float:
        length_norm = 1 - b + b * (self.doc_lengths[doc_id]/self.__get_avg_doc_length())
        tf = self.get_tf(doc_id, term)
        tf = (tf * (k1 + 1)) / (tf + k1 * length_norm)

        return tf

    def bm25(self, doc_id: int, term: str) -> float:
        idf = self.get_bm25_idf(term)
        tf = self.get_bm25_tf(doc_id, term)
        return idf * tf

    def bm25_search(self, query, limit) -> list[tuple[int, str, float]]:
        tokens = tokenize_text(query)
        scores: dict[int, float] = {}
        for doc_id in self.docmap:
            scores[doc_id] = 0
            for t in tokens:
                scores[doc_id] += self.bm25(doc_id, t)

        # .items method returns list of tuples, dict() turns tuple (x, y) into dict {x: y}
        # Our item below is a tuple (doc_id, score), so we sort by score `item[1]`
        sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]
        return [(doc_id, self.docmap[doc_id]["title"], score) for doc_id, score in sorted_scores]

    def build(self) -> None:
        movies = load_movies()
        for movie in movies:
            self.docmap[movie["id"]] = movie
            self.__add_documents(movie["id"],f"{movie['title']} {movie['description']}")

    def save(self) -> None:
        os.makedirs("cache", exist_ok = True)
        with open(INDEX_PICKLE_PATH, "wb") as idx_f:
            pickle.dump(self.index, idx_f)
        with open(DOCMAP_PICKLE_PATH, "wb") as doc_f:
            pickle.dump(self.docmap, doc_f)
        with open(TERM_FREQUENCIES_PICKLE_PATH, "wb") as term_freq_f:
            pickle.dump(self.term_frequencies, term_freq_f)
        with open(DOC_LENGTH_PICKLE_PATH, "wb") as doc_len_f:
            pickle.dump(self.doc_lengths, doc_len_f)

    def load(self) -> None:
        try:
            with open(INDEX_PICKLE_PATH, "rb") as idx_f:
                self.index = pickle.load(idx_f)
            with open(DOCMAP_PICKLE_PATH, "rb") as doc_f:
                self.docmap = pickle.load(doc_f)
            with open(TERM_FREQUENCIES_PICKLE_PATH, "rb") as term_freq_f:
                self.term_frequencies = pickle.load(term_freq_f)
            with open(DOC_LENGTH_PICKLE_PATH, "rb") as doc_len_f:
                self.doc_lengths = pickle.load(doc_len_f)
        except FileNotFoundError:
            return


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict] | None:
    idx = InvertedIndex()
    try:
        idx.load()
    except FileNotFoundError:
        print("Index not found, try 'build' command first")
        return
    seen: set[int] = set()
    results: list[dict] = []

    for token in tokenize_text(query):
        for doc_id in idx.get_documents(token):
            if doc_id not in seen:
                results.append(idx.docmap[doc_id])
                seen.add(doc_id)
            if len(results) >= limit:
                return results

def build_command() -> None:
    idx = InvertedIndex()
    idx.build()
    idx.save()

def stem(term: str) -> str:
    return PorterStemmer().stem(term.lower())

def tf_command(doc_id: int, term: str) -> int:
    idx = InvertedIndex()
    idx.load()
    return idx.get_tf(doc_id, term)

def idf_command(term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_idf(term)

def tfidf_command(doc_id: int, term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    idf = idx.get_idf(term)
    tf = idx.get_tf(doc_id, term)
    return tf * idf

def bm25_idf_command(term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    idf = idx.get_bm25_idf(term)
    return idf

def bm25_tf_command(doc_id: int, term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    tf = idx.get_bm25_tf(doc_id, term)
    return tf

def bm25_search_command(term: str, limit: int) -> list[tuple[int, str, float]]:
    idx = InvertedIndex()
    idx.load()
    bm25 = idx.bm25_search(term, limit)
    return bm25

def tokenize_text(text: str) -> list[str]:
    stripped = text.lower().translate(text.maketrans("", "", string.punctuation))
    tokens = [s for s in stripped.split() if s != ""]
    stop_words = load_stopwords()
    tokens = [q for q in tokens if q not in stop_words]
    return [stem(q) for q in tokens]
