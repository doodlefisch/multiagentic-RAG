# backend/pipeline.py (facultatif, si vous voulez un pipeline unifié)
from .loader import PDFLoader
from .chunker import DocumentChunker
from .embeddings import EmbeddingsEncoder
from .vectorstore import ChromaVectorStore

class IndexPipeline:
    def __init__(self, db_dir: str, collection_name: str):
        self.loader = PDFLoader()
        self.chunker = DocumentChunker()
        self.encoder = EmbeddingsEncoder()
        self.vectorstore = ChromaVectorStore(db_dir=db_dir, collection_name=collection_name)

    def process_pdf(self, file_path: str):
        # Charger
        docs = self.loader.load_pdf_as_documents(file_path)

        # Nettoyage optionnel
        # docs = [Document(page_content=self.loader.clean_text(doc.page_content), metadata=doc.metadata) for doc in docs]

        # Chunking
        chunked_docs = self.chunker.split_documents(docs)

        # Embeddings
        embeddings = self.encoder.encode_documents(chunked_docs)

        # Insertion dans Chroma
        self.vectorstore.insert_documents(chunked_docs, embeddings)
