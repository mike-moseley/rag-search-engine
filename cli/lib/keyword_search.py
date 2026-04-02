import pickle
import os
import string
from nltk.stem import PorterStemmer
from lib import search_utils


class InvertedIndex:
    def __init__(self):
        self.index = {}
        self.docmap = {}

    def __add_documents(self, doc_id, text):
        tokens = tokenize_text(text)
        for token in tokens:
            if token not in self.index:
                self.index[token] = {doc_id}
            else:
                self.index[token].add(doc_id)

    def get_documents(self, term):
        docs = self.index.get(term.lower(), set())
        return sorted(list(docs))

    def build(self):
        movies = search_utils.load_movies()
        for movie in movies:
            self.docmap[movie["id"]] = movie
            self.__add_documents(movie["id"],f"{movie['title']} {movie['description']}")

    def save(self):
        os.makedirs("cache", exist_ok = True)
        with open("cache/index.pkl", "wb") as f:
            pickle.dump(self.index, f)
        with open("cache/docmap.pkl", "wb") as f:
            pickle.dump(self.docmap, f)

def search_command(query, limit = search_utils.DEFAULT_SEARCH_LIMIT):
    movies = search_utils.load_movies()
    results = []
    movies = search_utils.load_movies()
    movies = tokenize_text(movies)

    for movie in movies:
        for query in tokenize_text(query):
            if any(query in token for token in movie):
                results.append(movie)
                if len(results) >= limit:
                    break
    return results

def build_command():
    idx = InvertedIndex()
    idx.build()
    idx.save()
    docs = idx.get_documents("merida")
    print(f"First document for token 'merida' = {docs[0]}")

def tokenize_text(str):
    stemmer = PorterStemmer()
    stripped = str.lower().translate(str.maketrans("", "", string.punctuation))
    tokens = [s for s in stripped.split(" ") if s != ""]
    stop_words = search_utils.load_stopwords()
    tokens = [q for q in tokens if q not in stop_words]
    return [stemmer.stem(q) for q in tokens ]
