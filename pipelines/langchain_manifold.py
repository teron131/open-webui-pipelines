import os
from typing import Generator, Iterator, List, Union

from langchain.schema import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_playground.UniversalChain import UniversalChain
from pydantic import BaseModel


class Pipeline:
    class Valves(BaseModel):
        OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
        AZURE_OPENAI_API_KEY: str = os.getenv("AZURE_OPENAI_API_KEY", "")
        AZURE_OPENAI_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        AZURE_OPENAI_DEPLOYMENT_NAME: str = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "")
        AZURE_OPENAI_API_VERSION: str = os.getenv("AZURE_OPENAI_API_VERSION", "")
        GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
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
            {"id": "gpt-4o-mini", "name": "GPT-4o-mini"},
            {"id": "gpt-4o-2024-11-20", "name": "GPT-4o"},
            {"id": "o1-mini", "name": "o1 mini"},
            {"id": "o1", "name": "o1"},
            {"id": "o3-mini", "name": "o3 mini"},
            {"id": "gemini-2.0-flash-exp", "name": "Gemini 2.0 Flash"},
            {"id": "gemini-2.0-flash-thinking-exp", "name": "Gemini 2.0 Flash Thinking"},
            {"id": "gemini-exp-1206", "name": "Gemini Experimental"},
            {"id": "claude-3.5-sonnet", "name": "Claude 3.5 Sonnet"},
            {"id": "deepseek-chat", "name": "DeepSeek V3"},
            {"id": "deepseek-deepseek-r1", "name": "DeepSeek R1"},
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
