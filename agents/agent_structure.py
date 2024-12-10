from .base_agent import BaseAgent

class AgentStructure(BaseAgent):
    def __init__(self, llm, structured_vectorstore):
        super().__init__(llm)
        self.structured_vectorstore = structured_vectorstore

    def run(self, query: str) -> str:
        # Recherche dans l’index structuré
        docs = self.structured_vectorstore.query_documents(query)
        # Synthèse par LLM
        prompt = f"Voici des passages structurés: {docs}\nRépond à la question : {query}"
        answer = self.llm(prompt)
        return answer
