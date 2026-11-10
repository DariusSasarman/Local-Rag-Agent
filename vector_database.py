import os
import sys
from datetime import datetime, timezone

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from encoder import get_encoder
import glob

class Vector_db ():
    def __init__(self):
        self.embeddings = get_encoder()
        self.vector_db = Chroma(
            persist_directory="./chroma_db",
            embedding_function=self.embeddings
        )

    def process_pdf(self,pdf_path : str, extra_metadata : dict | None = None):

        if not os.path.exists(pdf_path):
            return;

        reader = PdfReader(pdf_path)
        text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
        document = Document(page_content=text, metadata={"source": pdf_path})
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = text_splitter.split_documents([document])

        filename = os.path.basename(pdf_path)
        ingested_at = datetime.now(timezone.utc).isoformat()
        for i,chunk in enumerate(chunks):
            chunk.metadata.update({
                "filename": filename,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "ingested_at": ingested_at,
                **(extra_metadata or {}),
            })

        ids = [f"{filename}_{i}" for i in range(len(chunks))]

        self.vector_db.add_documents(documents=chunks, ids=ids)
    
    @staticmethod
    def find_ext(dr, ext, ig_case=False):
        if ig_case:
            ext = "".join([f"[{ch + ch.swapcase()}]" for ch in ext])
        return glob.glob(os.path.join(dr, "*." + ext))

    def process_folder(self, folder_path: str):
        if not os.path.exists(folder_path) or not os.listdir(folder_path):
            return

        files = self.find_ext(folder_path, "pdf")

        for file in files:
            self.process_pdf(file)

    def as_retriever(self, k: int = 4):
        return self.vector_db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": k * 4, "lambda_mult": 0.5},
        ) 

if __name__ == "__main__":
    target_dir = ""
    if len(sys.argv) > 1 :
        target_dir = sys.argv[1]
    else:
        target_dir = os.path.join(os.path.dirname(__file__), "documents_to_be_processed")

    print(f"Loading documents from \"{target_dir}\" into vector db")
    Vector_db().process_folder(target_dir)
    print("Loading executed successfully!")