import os
from typing import Generator, Iterator, List, Union

from pydantic import BaseModel


class Pipeline:
    class Valves(BaseModel):
        OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
        # ...
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
            # ...
        ]

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> Union[str, Generator, Iterator]:
        print(f"pipe:{__name__}")
        print(f"model_id: {model_id}")

        chain = self.get_chain()

        print(body)

        try:
            if body["stream"]:
                return (chunk.content for chunk in chain.stream(messages))
            else:
                return chain.invoke(messages).content
        except Exception as e:
            return f"Error: {e}"

    def get_chain(self):
        # ...
        return
