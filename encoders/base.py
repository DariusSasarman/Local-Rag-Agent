from abc import ABC, abstractmethod
from langchain_core.embeddings import Embeddings

class BaseEncoder(Embeddings, ABC):
    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        ...

    @classmethod
    @abstractmethod
    def get_model(cls):
        ...
    
    @classmethod
    @abstractmethod
    def get_encoder(cls):
        ...
