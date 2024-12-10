# backend/embeddings.py
from sentence_transformers import SentenceTransformer
from langchain.schema import Document

class EmbeddingsEncoder:
    def __init__(self, model_name: str='sentence-transformers/all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def encode_documents(self, chunks: list[Document]) -> list:
        texts = [doc.page_content for doc in chunks]
        embeddings = self.model.encode(texts, convert_to_tensor=False)
        return embeddings
