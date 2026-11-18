import os
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from encoders.base import BaseEncoder


class BGEEmbeddings(BaseEncoder):
    def __init__(self):
        self.model = self.get_model()

    def embed_documents(self, texts):
        return self.model.encode(
            texts, normalize_embeddings=True, batch_size=8
        ).tolist()

    def embed_query(self, text):
        return self.model.encode(
            text,
            normalize_embeddings=True
        ).tolist()

    @classmethod
    @lru_cache(maxsize=1)
    def get_model(cls):
        MODEL_NAME = "BAAI/bge-m3"
        LOCAL_PATH = "./models/local_bge_m3"
        if not os.path.exists(LOCAL_PATH) or not os.listdir(LOCAL_PATH):
            os.makedirs(LOCAL_PATH, exist_ok=True)
            model = SentenceTransformer(MODEL_NAME, device="cpu")
            model.save(LOCAL_PATH)
            return model
        return SentenceTransformer(LOCAL_PATH, device="cpu")

    @classmethod
    def get_encoder(cls):
        return BGEEmbeddings()
