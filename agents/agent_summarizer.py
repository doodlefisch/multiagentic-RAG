from .base_agent import BaseAgent

class AgentSummarizer(BaseAgent):
    def run(self, structured_answer: str, nonstructured_answer: str, user_query: str) -> str:
        prompt = f"L'utilisateur a demandé : {user_query}\n\nRéponse du système structuré : {structured_answer}\n\nRéponse du système non structuré : {nonstructured_answer}\n\nFais une synthèse et une réponse complète au user."
        summary = self.llm(prompt)
        return summary
