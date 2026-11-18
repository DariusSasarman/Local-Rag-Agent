import hashlib
import os
from collections import defaultdict

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from retrievers.base import Retriever
from retrievers.bm25s_retriever import BM25SRetriever
from retrievers.vector_retriever import VectorRetriever
from models.cross_encoding_reranker import LocalReranker

K_RRF = 60

class HybridRetriever (Retriever):

    def __init__(self, save_location : str = "./hybrid"):
        vec_store_location = os.path.join(save_location,"vec")
        key_store_location = os.path.join(save_location,"key")

        if not os.path.exists(save_location) or not os.listdir(save_location):
            os.makedirs(save_location, exist_ok=True)
            os.makedirs(vec_store_location, exist_ok=True)
            os.makedirs(key_store_location,exist_ok=True)

        self.vec_ret = VectorRetriever(vec_store_location)
        self.key_ret = BM25SRetriever(key_store_location)
        self.reranker = LocalReranker()


    def save_document(self, documents: list[Document], file_path: str):
        self.vec_ret.save_document(documents, file_path)
        self.key_ret.save_document(documents, file_path)

    def as_retriever(self, k: int = 4, fetch_k: int = 20):
        retrievers = [
            self.vec_ret.as_retriever(k=fetch_k),
            self.key_ret.as_retriever(k=fetch_k),
        ]
        fuse = self._reciprocal_rank_fusion
        reranker = self.reranker

        class _Hybrid(BaseRetriever):
            def _get_relevant_documents(
                    self, query: str, *, run_manager: CallbackManagerForRetrieverRun
            ) -> list[Document]:
                docs, rankings = {}, []
                for r in retrievers:
                    ranking = []
                    for d in r.invoke(query):
                        key = d.metadata.get("id") or hashlib.sha1(
                            d.page_content.encode()
                        ).hexdigest()
                        docs.setdefault(key, d)
                        ranking.append(key)
                    rankings.append(ranking)

                # 1. RRF -> candidates
                fused = fuse(rankings)[: fetch_k * 2]
                keys = [key for key, _ in fused]

                # 2. Cross-encoder on the candidate texts
                texts = [docs[key].page_content for key in keys]
                ranked = reranker.rerank(query, texts, top_n=k)

                # 3. Map indices back to documents
                results = []
                for idx, score in ranked:
                    doc = docs[keys[idx]]
                    doc.metadata["rerank_score"] = score
                    results.append(doc)
                return results[:k]

        return _Hybrid()

    @staticmethod
    def _reciprocal_rank_fusion( rankings: list[list[str]], k: int = K_RRF) -> list[tuple[str, float]]:
        """Fuse multiple ranked lists of doc_ids into one ranked list."""
        scores: dict[str, float] = defaultdict(float)
        for ranking in rankings:
            for rank, doc_id in enumerate(ranking, start=1):
                scores[doc_id] += 1.0 / (k + rank)
        return sorted(scores.items(), key=lambda x: -x[1])