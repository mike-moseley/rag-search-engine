import pickle
import os
from keyword_search_cli import tokenize,load_movies
class InvertedIndex:
    def __init__(self):
        self.index = {}
        self.docmap = {}

    def __add_documents(self, doc_id, text):
        tokens = tokenize(text)
        for token in tokens:
            if token not in self.index:
                self.index[token] = {doc_id}
            else:
                self.index[token].add(doc_id)

    def get_documents(self, term):
        docs = self.index.get(term.lower(), set())
        return sorted(list(docs))

    def build(self):
        movies = load_movies()
        for movie in movies:
            self.docmap[movie["id"]] = movie
            self.__add_documents(movie["id"],f"{movie['title']} {movie['description']}")

    def save(self):
        os.makedirs("cache", exist_ok = True)
        with open("cache/index.pkl", "wb") as f:
            pickle.dump(self.index, f)
        with open("cache/docmap.pkl", "wb") as f:
            pickle.dump(self.docmap, f)
