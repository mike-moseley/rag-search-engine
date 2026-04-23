from PIL import Image
from sentence_transformers import SentenceTransformer
from .search_utils import load_movies
from .semantic_search import cosine_similarity

class MultimodalSearch:
    def __init__(self, documents: list[dict], model_name="clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)
        self.documents = documents
        self.texts = []
        for movie in self.documents:
            self.texts.append(f"{movie['title']}: {movie['description']}")
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)

    def embed_image(self, path: str):
        img = Image.open(path)
        return self.model.encode([img])[0] # type: ignore

    def search_with_image(self, path:str):
        embedding = self.embed_image(path)
        scores = []
        for i,text in enumerate(self.text_embeddings):
            result = {}
            movie = self.documents[i]
            result["doc_id"] = movie["id"]
            result["title"] = movie["title"]
            result["description"] = movie["description"]
            result["score"] = cosine_similarity(text, embedding)
            scores.append(result)
        return sorted(scores, key=lambda x: x["score"], reverse=True)

def verify_image_embedding(path: str):
    movies = load_movies()
    mms = MultimodalSearch(movies)
    embedding = mms.embed_image(path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")

def image_search_command(path: str):
    movies = load_movies()
    mms = MultimodalSearch(movies)
    results = mms.search_with_image(path)
    return results
