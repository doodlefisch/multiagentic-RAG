from .base_agent import BaseAgent

class AgentOrchestrateur10K:
    def __init__(self, agent_structure, agent_nonstructure, agent_summarizer):
        self.agent_structure = agent_structure
        self.agent_nonstructure = agent_nonstructure
        self.agent_summarizer = agent_summarizer

    def run(self, user_query: str) -> str:
        # Exemples de prompts internes :
        financial_data = self.agent_structure.run("Données financières clés du 10-K")
        risk_data = self.agent_structure.run("Risques majeurs identifiés dans le 10-K")
        additional_context = self.agent_nonstructure.run("Résumé des sections importantes du rapport")

        # Combiner tout avec summarizer
        final_answer = self.agent_summarizer.run(
            structured_answer=financial_data, 
            nonstructured_answer=additional_context, 
            user_query=user_query
        )
        return final_answer
