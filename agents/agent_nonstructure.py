from .base_agent import BaseAgent

class AgentNonStructure(BaseAgent):
    def __init__(self, llm, nonstructured_vectorstore):
        super().__init__(llm)
        self.nonstructured_vectorstore = nonstructured_vectorstore

    def run(self, query: str) -> str:
        docs = self.nonstructured_vectorstore.query_documents(query)
        prompt = f"Voici des passages non structurés : {docs}\nRépond à la question : {query}"
        answer = self.llm(prompt)
        return answer
