import os
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings

MODEL_NAME = "BAAI/bge-m3"
LOCAL_PATH = "./models/local_bge_m3"


@lru_cache(maxsize=1)
def get_model():
    if not os.path.exists(LOCAL_PATH) or not os.listdir(LOCAL_PATH):
        os.makedirs(LOCAL_PATH, exist_ok=True)
        model = SentenceTransformer(MODEL_NAME, device="cpu")
        model.save(LOCAL_PATH)
        return model
    return SentenceTransformer(LOCAL_PATH, device="cpu")


class BGEEmbeddings(Embeddings):
    def __init__(self):
        self.model = get_model()

    def embed_documents(self, texts):
        return self.model.encode(
            texts, normalize_embeddings=True, batch_size=8
        ).tolist()

    def embed_query(self, text):
        return self.model.encode(
            "Represent this sentence for searching relevant passages: " + text,
            normalize_embeddings=True
        ).tolist()


def get_encoder():
    return BGEEmbeddings()