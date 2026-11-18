import hashlib
import os
import sqlite3
import sys
from typing import Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

from encoders.best_matching_25_sparse import BM25Embeddings
from retrievers.base import Retriever

class _BM25SSearch(BaseRetriever):
    index: Any
    docs: list[Document]
    model: Any
    k: int = 4

    def _get_relevant_documents(self, query: str, *, run_manager=None):
        if not self.docs:
            return []
        tokens = self.model.tokenize([query], stopwords="en", return_ids=False)
        results, _ = self.index.retrieve(
            tokens, corpus=self.docs, k=min(self.k, len(self.docs))
        )
        return list(results[0])

class BM25SRetriever(Retriever):

    def __init__(self, db_path: str = "./bm25.db"):
        self.encoder = BM25Embeddings.get_encoder()
        if not os.path.exists(db_path) or not os.listdir(db_path):
            os.makedirs(db_path, exist_ok=True)
        self.db = sqlite3.connect(db_path)
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                filepath TEXT,
                chunk_index INTEGER,
                content TEXT
            )
        """)

    def save_document(self, documents: list[Document], file_path: str):
        text = "\n\n".join(d.page_content for d in documents)
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_text(text)

        path = os.path.abspath(file_path)
        key = hashlib.sha1(path.encode()).hexdigest()[:12]

        with self.db:
            self.db.execute("DELETE FROM chunks WHERE filepath = ?", (path,))
            self.db.executemany(
                "INSERT INTO chunks VALUES (?, ?, ?, ?)",
                [(f"{key}_{i}", path, i, c) for i, c in enumerate(chunks)],
            )

    def as_retriever(self, k: int = 4):
        rows = self.db.execute(
            "SELECT filepath, chunk_index, content FROM chunks"
        ).fetchall()
        docs = [
            Document(
                page_content=content,
                metadata={
                    "filepath": path,
                    "filename": os.path.basename(path),
                    "chunk_index": i,
                },
            )
            for path, i, content in rows
        ]

        index = self.encoder.model.BM25()
        if docs:
            index.index(self.encoder.embed_documents([d.page_content for d in docs]))

        return _BM25SSearch(index=index, docs=docs, model=self.encoder.model, k=k)

if __name__ == "__main__":
    BM25SRetriever().run_as_main(sys.argv)