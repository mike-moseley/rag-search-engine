import os
import json

DEFAULT_SEARCH_LIMIT = 5

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")

def load_movies():
    with open(DATA_PATH, "r") as f:
        return json.load(f)["movies"]

def load_stopwords():
    with open(STOPWORDS_PATH, "r") as f:
        return f.read().splitlines()
