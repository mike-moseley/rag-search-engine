import os
import json

from lib.consts import DATA_PATH, STOPWORDS_PATH
def load_movies() -> list[dict]:
    with open(DATA_PATH, "r") as f:
        return json.load(f)["movies"]

def load_stopwords() -> list[str]:
    with open(STOPWORDS_PATH, "r") as f:
        return f.read().splitlines()
