from .loader import load_pdf_as_documents
from .chunker import split_documents
from .embeddings import encode_documents
from .vectorstore import init_chroma_db, insert_documents, query_documents

def main():
    # Chemin vers le fichier PDF à tester
    pdf_path = "C:/Users/romai/OneDrive/Pièces jointes/Documents/ESILV A5/pi3/data/dixK/apple.pdf"

    # 1. Charger les documents depuis le fichier PDF
    print("Chargement des documents...")
    documents = load_pdf_as_documents(pdf_path)
    print(f"Nombre de documents chargés : {len(documents)}")

    # 2. Diviser les documents en chunks
    print("Chunking des documents...")
    chunked_documents = split_documents(documents)
    print(f"Nombre de chunks créés : {len(chunked_documents)}")

    # 3. Encoder les chunks en embeddings
    print("Encodage des documents en embeddings...")
    embeddings = encode_documents(chunked_documents)
    print(f"Nombre d'embeddings : {len(embeddings)}")
    print(f"Dimension des embeddings : {len(embeddings[0])}")
    # 4. Initialiser la base ChromaDB
    print("Initialisation de la base ChromaDB...")
    collection = init_chroma_db(db_dir="./my_chroma_db")
    # 5. Insérer les chunks et embeddings dans ChromaDB
    # On récupère juste le texte des chunks pour l'insertion
    chunk_texts = [doc.page_content for doc in chunked_documents]
    print("Insertion des documents dans ChromaDB...")
    insert_documents(collection, chunk_texts, embeddings)
    print("Insertion terminée.")

    # 6. Interroger la base vectorielle
    query = "Quels sont les meilleures ventes de Apple cette année ?"
    print(f"Interrogation de la base vectorielle avec la requête : '{query}'")
    results = query_documents(collection, query, top_k=5)
    print("Résultats de la requête :")
    for i, (doc, score) in enumerate(zip(results['documents'][0], results['distances'][0])):
        print(f"Résultat {i+1}:")
        print("Texte du chunk :", doc)
        print("Score (distance) :", score)
        print("------")

if __name__ == "__main__":
    main()
