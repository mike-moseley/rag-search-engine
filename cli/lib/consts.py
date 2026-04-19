import os

DEFAULT_SEARCH_LIMIT = 5

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
GOLDEN_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "golden_dataset.json")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")
INDEX_PICKLE_PATH = os.path.join(PROJECT_ROOT, "cache", "index.pkl")
DOCMAP_PICKLE_PATH = os.path.join(PROJECT_ROOT, "cache", "docmap.pkl")
TERM_FREQUENCIES_PICKLE_PATH = os.path.join(PROJECT_ROOT, "cache", "term_frequencies.pkl")
DOC_LENGTH_PICKLE_PATH = os.path.join(PROJECT_ROOT, "cache", "doc_lengths.pkl")

EMBEDDING_PATH = os.path.join(PROJECT_ROOT, "cache", "movie_embeddings.npy")
CHUNK_EMBEDDING_PATH = os.path.join(PROJECT_ROOT, "cache", "chunk_embeddings.npy")
CHUNK_METADATA_PATH = os.path.join(PROJECT_ROOT, "cache", "chunk_metadata.json")
SCORE_PRECISION = 3
BM25_K1 = 1.5
BM25_B = 0.75
