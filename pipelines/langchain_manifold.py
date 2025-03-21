import os
from typing import Generator, Iterator, List, Union

from langchain_playground.universal import UniversalChain
from pydantic import BaseModel


class Pipeline:
    class Valves(BaseModel):
        OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
        GOOGLE_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
        OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
        LANGCHAIN_TRACING_V2: bool = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
        LANGCHAIN_API_KEY: str = os.getenv("LANGCHAIN_API_KEY", "")
        LANGCHAIN_PROJECT: str = os.getenv("LANGCHAIN_PROJECT", "")
        pass

    def __init__(self):
        self.type = "manifold"
        self.name = "LangChain: "
        self.valves = self.Valves()
        self.pipelines = self.get_models()
        pass

    async def on_startup(self):
        print(f"on_startup:{__name__}")
        pass

    async def on_shutdown(self):
        print(f"on_shutdown:{__name__}")
        pass

    def get_models(self):
        return [
            {"id": "openai/gpt-4o-mini", "name": "GPT-4o-mini"},
            {"id": "openai/o3-mini", "name": "o3 mini"},
            {"id": "google/gemini-2.0-flash-001", "name": "Gemini 2.0 Flash"},
            {"id": "google/gemini-2.0-flash-thinking-exp:free", "name": "Gemini 2.0 Flash Thinking"},
            {"id": "google/gemini-2.0-pro-exp-02-05:free", "name": "Gemini 2.0 Pro"},
            {"id": "anthropic/claude-3.7-sonnet", "name": "Claude 3.7 Sonnet"},
            {"id": "anthropic/claude-3.7-sonnet:thinking", "name": "Claude 3.7 Sonnet Thinking"},
        ]

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> Union[str, Generator, Iterator]:
        """Generate a response using a language model chain.

        Args:
            user_message (str): The input message from the user
            model_id (str): ID of the language model to use
            messages (List[dict]): List of conversation messages
            body (dict): Additional request parameters

        Returns:
            Union[str, Generator, Iterator]: Generated response or stream of responses
        """
        print(f"pipe:{__name__}")
        print(f"model_id: {model_id}")
        print(body)

        try:
            # Convert UI messages to LangChain format
            lc_messages = [(msg["role"], msg["content"]) for msg in messages if msg["role"] in ["user", "assistant"]]

            chain = UniversalChain(model_id=model_id)

            return chain.invoke(user_message, message_history=lc_messages)

        except Exception as e:
            return f"Error: {e}"
