from abc import ABC, abstractmethod
import glob
import os
from langchain_core.documents import Document
from langchain_unstructured import UnstructuredLoader

class Retriever(ABC):

    @abstractmethod
    def save_document(self, documents: list[Document], file_path: str):
        """
        Saves the loaded elements of one file to the datastore.
        The implementation decides how to merge/chunk them.
        :param documents: All elements loaded from the file
        :param file_path: Path to the source file
        """
        ...

    @abstractmethod
    def as_retriever(self, k: int = 4):
        ...

    def ingest_document(self, file_path: str):
        if not os.path.isfile(file_path):
            return
        documents = UnstructuredLoader(file_path).load()
        if documents:
            self.save_document(documents, file_path)

    def process_folder(self, folder_path: str):
        if not os.path.isdir(folder_path):
            return
        pattern = os.path.join(folder_path, "**", "*")
        for file in glob.glob(pattern, recursive=True):
            self.ingest_document(file)

    def run_as_main(self, args):
        if len(args) < 2:
            print("Please input a target folder path.")
            return
        target_dir = args[1]
        print(f'Loading documents from "{target_dir}" into vector db')
        self.process_folder(target_dir)
        print("Loading executed successfully!")