import bm25s
from encoders.base import BaseEncoder

class BM25Embeddings(BaseEncoder):
    def __init__(self):
        self.model = self.get_model()

    def embed_documents(self, texts: list[str], stopwords: str = "en"):
        return self.model.tokenize(texts, stopwords=stopwords)

    def embed_query(self, text: str, stopwords: str = "en"):
        return self.model.tokenize([text], stopwords=stopwords)

    @classmethod
    def get_model(cls):
        return bm25s

    @classmethod
    def get_encoder(cls):
        return BM25Embeddings()
