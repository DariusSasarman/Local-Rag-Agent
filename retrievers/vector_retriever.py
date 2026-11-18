import hashlib
import os
import sys
from datetime import datetime, timezone
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from encoders.vector_encoder import BGEEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores.utils import filter_complex_metadata
from retrievers.base import Retriever

class VectorRetriever(Retriever):

    def __init__(self, persist_directory: str = "./vector_db"):
        self.embeddings = BGEEmbeddings.get_encoder()
        if not os.path.exists(persist_directory) or not os.listdir(persist_directory):
            os.makedirs(persist_directory, exist_ok=True)

        self.vector_db = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings,
        )

    def as_retriever(self, k: int = 4):
        return self.vector_db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": k * 4, "lambda_mult": 0.5},
        )

    def save_document(self, documents: list[Document], file_path: str):
        # Merge all elements into one text, then chunk it
        text = "\n\n".join(d.page_content for d in documents)
        merged = Document(page_content=text, metadata={"filepath": file_path})

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents([merged])

        filename = os.path.basename(file_path)
        path_hash = hashlib.sha1(os.path.abspath(file_path).encode()).hexdigest()[:12]
        ingested_at = datetime.now(timezone.utc).isoformat()

        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "filename": filename,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "ingested_at": ingested_at,
            })

        chunks = filter_complex_metadata(chunks)
        ids = [f"{path_hash}_{i}" for i in range(len(chunks))]
        self.vector_db.add_documents(documents=chunks, ids=ids)


if __name__ == "__main__":
    VectorRetriever().run_as_main(sys.argv)