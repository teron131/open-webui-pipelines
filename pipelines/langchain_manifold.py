import os
from typing import Generator, Iterator, List, Union

from langchain.chat_models.base import init_chat_model
from pydantic import BaseModel


class Pipeline:
    class Valves(BaseModel):
        OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
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
            {"id": "claude-3.5-sonnet", "name": "Claude 3.5 Sonnet"},
        ]

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> Union[str, Generator, Iterator]:
        print(f"pipe:{__name__}")
        print(f"model_id: {model_id}")
        if "claude" in model_id:
            llm = init_chat_model(
                model=f"anthropic/{model_id}",  # Avoid making model_id with '/', otherwise it will mess up the FastAPI URL
                model_provider="openai",
                base_url="https://openrouter.ai/api/v1",
                api_key=os.getenv("OPENROUTER_API_KEY", ""),
            )
        else:
            llm = init_chat_model(model=model_id)

        chain = llm

        print(body)

        try:
            if body["stream"]:
                return (chunk.content for chunk in chain.stream(messages))
            else:
                return chain.invoke(messages).content
        except Exception as e:
            return f"Error: {e}"
