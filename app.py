# app.py
import streamlit as st
from dotenv import load_dotenv
from agents.agent_structure import AgentStructure
from agents.agent_nonstructure import AgentNonStructure
from agents.agent_summarizer import AgentSummarizer
from agents.agent_orchestrator_tenK import AgentOrchestrateur10K
from agents.agent_orchestrator import AgentOrchestrateurPrincipal

from backend.pipeline import IndexPipeline
from backend.vectorstore import ChromaVectorStore
from backend.loader import PDFLoader
from backend.chunker import DocumentChunker
from backend.embeddings import EmbeddingsEncoder
from backend.vectorstore import ChromaVectorStore

from LLM_Manager import LLMManager  # supposez que vous avez mis la classe LLMManager dans un fichier LLMManager_file.py
load_dotenv()

llm_manager = LLMManager()  # on instancie la classe LLMManager
# On définit une fonction llm qui pointe vers generate_response pour compatibilité
def llm(prompt):
    return llm_manager.generate_response(prompt)

st.title("Multi-Agents RAG")

# Indexation du 10-K
st.header("Indexation 10-K")
uploaded_10k = st.file_uploader("Upload 10-K PDF", type=["pdf"], key="tenk")
if uploaded_10k is not None:
    if st.button("Indexer 10-K"):
        pipeline = IndexPipeline(
            db_dir="data/tenk_index",
            collection_name="tenk_collection"
        )
        pdf_path = f"data/tenk_index/{uploaded_10k.name}"
        with open(pdf_path, "wb") as f:
            f.write(uploaded_10k.getbuffer())
        pipeline.process_pdf(pdf_path)
        st.success("10-K indexé !")
        # On stocke le vectorstore dans la session
        st.session_state["tenk_vectorstore"] = pipeline.vectorstore

# Indexation docs génériques
st.header("Indexation documents génériques")
uploaded_doc = st.file_uploader("Upload autre PDF", type=["pdf"], key="generic")
if uploaded_doc is not None:
    if st.button("Indexer document générique"):
        pipeline = IndexPipeline(
            db_dir="data/generic_index",
            collection_name="generic_collection"
        )
        pdf_path = f"data/generic_index/{uploaded_doc.name}"
        with open(pdf_path, "wb") as f:
            f.write(uploaded_doc.getbuffer())
        pipeline.process_pdf(pdf_path)
        st.success("Document générique indexé !")
        st.session_state["generic_vectorstore"] = pipeline.vectorstore

# Chatbot
st.header("Chatbot")
user_query = st.text_input("Posez votre question:")
if st.button("Envoyer"):
    if "tenk_vectorstore" in st.session_state and "generic_vectorstore" in st.session_state:
        agent_structure = AgentStructure(llm=llm, structured_vectorstore=st.session_state["tenk_vectorstore"])
        agent_nonstructure = AgentNonStructure(llm=llm, nonstructured_vectorstore=st.session_state["generic_vectorstore"])
        agent_summarizer = AgentSummarizer(llm=llm)
        agent_orchestrateur_10k = AgentOrchestrateur10K(
            agent_structure=agent_structure,
            agent_nonstructure=agent_nonstructure,
            agent_summarizer=agent_summarizer
        )
        agent_principal = AgentOrchestrateurPrincipal(
            agent_structure=agent_structure,
            agent_nonstructure=agent_nonstructure,
            agent_summarizer=agent_summarizer,
            agent_orchestrateur_10k=agent_orchestrateur_10k
        )

        answer = agent_principal.run(user_query)
        st.write(answer)
    else:
        st.warning("Veuillez indexer au moins un 10-K et un document générique avant de poser une question.")
