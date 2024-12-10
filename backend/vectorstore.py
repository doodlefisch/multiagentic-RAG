import chromadb
from chromadb.config import Settings

def init_chroma_db(db_dir: str):
    # Utilisez PersistentClient au lieu de Client
    client = chromadb.PersistentClient(path=db_dir)
    # Créez ou récupérez la collection
    collection = client.get_or_create_collection(name="tenk_reports")
    return collection


def insert_documents(collection, chunks, embeddings):
    docs_content = [doc.page_content for doc in chunks]
    ids = [f"doc_{i}" for i in range(len(chunks))]
    collection.add(documents=docs_content, embeddings=embeddings, ids=ids)

def query_documents(collection, query, top_k=5):
    results = collection.query(query_texts=[query], n_results=top_k)
    return results

# backend/vectorstore.py
import chromadb
from chromadb.config import Settings

class ChromaVectorStore:
    def __init__(self, db_dir: str, collection_name: str="tenk_reports"):
        # Utilisez PersistentClient si disponible, sinon Client standard.
        # En fonction de la version de Chroma, PersistentClient peut être remplacé par:
        # client = chromadb.PersistentClient(path=db_dir)
        # Si non disponible, utiliser:
        # client = chromadb.Client(Settings(persist_directory=db_dir, chroma_db_impl="duckdb+parquet"))
        
        # On part du principe que PersistentClient fonctionne comme indiqué dans le code fourni.
        try:
            # Tentative avec PersistentClient (si votre version ChromaDB le permet)
            client = chromadb.PersistentClient(path=db_dir)
        except AttributeError:
            # Fallback si PersistentClient n'est pas dispo
            client = chromadb.Client(Settings(
                persist_directory=db_dir,
                chroma_db_impl="duckdb+parquet"
            ))
        
        self.collection = client.get_or_create_collection(name=collection_name)

    def insert_documents(self, chunks, embeddings):
        docs_content = [doc.page_content for doc in chunks]
        ids = [f"doc_{i}" for i in range(len(chunks))]
        self.collection.add(documents=docs_content, embeddings=embeddings, ids=ids)


    def query_documents(self, query: str, top_k=5):
        results = self.collection.query(query_texts=[query], n_results=top_k)
        return results
