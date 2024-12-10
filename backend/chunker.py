# backend/chunker.py
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocumentChunker:
    def __init__(self, chunk_size=2000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators
        )

    def split_documents(self, documents: list[Document]) -> list[Document]:
        # Cette opération crée une liste de documents chunkés
        chunked_docs = self.text_splitter.split_documents(documents)
        return chunked_docs
