from typing import List, Union, Generator, Iterator
from pydantic import BaseModel
import os
import requests


class Pipeline:
    class Valves(BaseModel):
        BOTBUILDER_API_KEY: str = os.getenv("BOTBUILDER_API_KEY", "your-botbuilder-api-key-here")
        BOTBUILDER_GROUP_ID: str = os.getenv("BOTBUILDER_GROUP_ID", "your-botbuilder-group-id-here")
        BOTBUILDER_MODEL: str = os.getenv("BOTBUILDER_MODEL", "gpt-4o")
        pass

    def __init__(self):
        # Optionally, you can set the id and name of the pipeline.
        # Best practice is to not specify the id so that it can be automatically inferred from the filename, so that users can install multiple versions of the same pipeline.
        # The identifier must be unique across all pipelines.
        # The identifier must be an alphanumeric string that can include underscores or hyphens. It cannot contain spaces, special characters, slashes, or backslashes.
        # self.id = "botbuilder_pipeline"
        self.name = "Bot Builder"
        self.valves = self.Valves()
        pass

    async def on_startup(self):
        # This function is called when the server is started.
        print(f"on_startup:{__name__}")
        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        print(f"on_shutdown:{__name__}")
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom pipelines like RAG.
        print(f"pipe:{__name__}")

        print(messages)
        print(user_message)

        BOTBUILDER_API_KEY = self.valves.BOTBUILDER_API_KEY
        BOTBUILDER_GROUP_ID = self.valves.BOTBUILDER_GROUP_ID
        BOTBUILDER_MODEL = self.valves.BOTBUILDER_MODEL

        headers = {}
        headers["x-api-key"] = BOTBUILDER_API_KEY

        allowed_params = {"max_tokens", "temperature", "top", "top_p", "presence_penalty", "frequency_penalty", "prompt_template"}
        filtered_overrides = {k: v for k, v in body.items() if k in allowed_params}

        payload = {
            "approach": "rrr",
            "history": messages,
            "overrides": {
                "model": BOTBUILDER_MODEL,
                **filtered_overrides
            },
        }

        # Remove unnecessary fields
        payload.pop("user", None)
        payload.pop("chat_id", None)
        payload.pop("title", None)

        print(payload)

        try:
            r = requests.post(
                url=f"https://api.uat.bot-builder.pccw.com/v1/llm-models/{BOTBUILDER_GROUP_ID}/chat",
                json=payload,
                headers=headers,
                verify=True,
            )
            print(f"response: {r}")
            r.raise_for_status()
            json_response = r.json()

            # if body["stream"]:
            #     return r.iter_lines()
            # else:
            return json_response.get("message", {}).get("content", "")
        except Exception as e:
            return f"Error: {e}"
