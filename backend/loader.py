# backend/loader.py
from langchain.schema import Document
import PyPDF2
import unicodedata

class PDFLoader:
    def load_pdf_as_documents(self, filepath: str) -> list[Document]:
        documents = []
        with open(filepath, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                doc = Document(page_content=page_text, metadata={"source": filepath, "page": i})
                documents.append(doc)
        return documents

    def clean_text(self, text: str) -> str:
        # Supprimer les caractères de contrôle et non assignés
        text = ''.join(c for c in text if unicodedata.category(c)[0] != 'C')
        return text.strip()





