import os
from dotenv import load_dotenv
import requests

class LLMManager:
    """
    A class to manage interactions with the Groq LLM API.
    """
    def __init__(self):
        load_dotenv()  # Pour s'assurer que les variables d'env sont chargées
        self.api_key = os.getenv("GROQ_API_KEY")
        self.model = os.getenv("MODEL_NAME")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables.")
        self.endpoint = "https://api.groq.com/generate"

    def generate_response(self, prompt):
        """
        Use the LLM to generate a response based on the prompt.
        """
        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = requests.post(self.endpoint, headers=headers, json={"prompt": prompt})
        response.raise_for_status()
        return response.json()["response"]
