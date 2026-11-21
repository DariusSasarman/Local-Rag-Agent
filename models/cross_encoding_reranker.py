import gc
import os
from typing import Any

import torch
from pydantic import BaseModel, PrivateAttr
from sentence_transformers import CrossEncoder

MODEL_NAME = "cross-encoder/ettin-reranker-17m-v1"
MODELS_DIR = ""


def download_model():
    local_path = os.path.join(MODELS_DIR, MODEL_NAME.split("/")[1])

    if not os.path.exists(local_path) or not os.listdir(local_path):
        model = CrossEncoder(MODEL_NAME)
        model.save(local_path)
    else:
        model = CrossEncoder(local_path)

    return model


class LocalReranker(BaseModel):
    _model: Any = PrivateAttr(default=None)

    def load(self):
        if self._model is None:
            self._model = download_model()

    def unload(self):
        self._model = None
        gc.collect()
        torch.cuda.empty_cache()

    def rerank(self, query: str, documents: list[str], top_n: int = 10):
        """Returns [(index_in_documents, score), ...], the best first."""
        self.load()
        ranked = self._model.rank(query, documents, top_k=top_n)
        self.unload()
        return [(r["corpus_id"], float(r["score"])) for r in ranked]