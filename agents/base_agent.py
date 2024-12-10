class BaseAgent:
    def __init__(self, llm):
        self.llm = llm

    def run(self, query: str) -> str:
        raise NotImplementedError
