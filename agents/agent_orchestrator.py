from .base_agent import BaseAgent
class AgentOrchestrateurPrincipal:
    def __init__(self, agent_structure, agent_nonstructure, agent_summarizer, agent_orchestrateur_10k):
        self.agent_structure = agent_structure
        self.agent_nonstructure = agent_nonstructure
        self.agent_summarizer = agent_summarizer
        self.agent_orchestrateur_10k = agent_orchestrateur_10k

    def run(self, user_query: str) -> str:
        # Détection simplifiée : si le prompt contient "10-K"
        if "10-K" in user_query or "rapport 10k" in user_query.lower():
            return self.agent_orchestrateur_tenK.run(user_query)
        else:
            # Cas général : on interroge les deux agents puis summarizer
            structured_answer = self.agent_structure.run(user_query)
            nonstructured_answer = self.agent_nonstructure.run(user_query)
            return self.agent_summarizer.run(structured_answer, nonstructured_answer, user_query)
