import os
from typing import Generator, Iterator, List, Union

from langchain.chat_models.base import init_chat_model
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_openai.chat_models.azure import AzureChatOpenAI
from langchain_openai.chat_models.base import ChatOpenAI
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
            {"id": "gpt-4o", "name": "GPT-4o"},
            {"id": "gpt-4o-mini", "name": "GPT-4o-mini"},
            {"id": self.valves.AZURE_OPENAI_DEPLOYMENT_NAME, "name": self.valves.AZURE_OPENAI_DEPLOYMENT_NAME},
            {"id": "gemini-1.5-pro", "name": "Gemini 1.5 Pro"},
            {"id": "gemini-1.5-flash", "name": "Gemini 1.5 Flash"},
            {"id": "claude-3.5-sonnet", "name": "Claude 3.5 Sonnet"},
        ]

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> Union[str, Generator, Iterator]:
        print(f"pipe:{__name__}")
        print(f"model_id: {model_id}")

        llm = self.get_llm(model_id)
        chain = llm

        print(body)

        try:
            if body["stream"]:
                return (chunk.content for chunk in chain.stream(messages))
            else:
                return chain.invoke(messages).content
        except Exception as e:
            return f"Error: {e}"

    def get_llm(self, model_id: str):
        if "azure" in model_id:
            llm = AzureChatOpenAI(
                api_key=self.valves.AZURE_OPENAI_API_KEY,
                azure_endpoint=self.valves.AZURE_OPENAI_ENDPOINT,
                azure_deployment=self.valves.AZURE_OPENAI_DEPLOYMENT_NAME,
                api_version=self.valves.AZURE_OPENAI_API_VERSION,
            )
        elif "gemini" in model_id:
            llm = ChatGoogleGenerativeAI(model=model_id, api_key=self.valves.GEMINI_API_KEY)
        elif "claude" in model_id:
            llm = ChatOpenAI(
                model=f"anthropic/{model_id}",  # Avoid making model_id with '/', otherwise it will mess up the FastAPI URL
                base_url="https://openrouter.ai/api/v1",
                api_key=self.valves.OPENROUTER_API_KEY,
            )
        elif "gpt" in model_id:
            llm = ChatOpenAI(model=model_id, api_key=self.valves.OPENAI_API_KEY)
        else:
            raise ValueError(f"Invalid model_id: {model_id}")
        return llm
