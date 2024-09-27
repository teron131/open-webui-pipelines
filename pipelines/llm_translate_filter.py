import os
import re
from typing import List, Optional

import requests
from pydantic import BaseModel
from utils.pipelines.main import get_last_assistant_message, get_last_user_message


class Pipeline:
    class Valves(BaseModel):
        # List target pipeline ids (models) that this filter will be connected to.
        # If you want to connect this filter to all pipelines, you can set pipelines to ["*"]
        # e.g. ["llama3:latest", "gpt-3.5-turbo"]
        pipelines: List[str] = ["*"]

        # Assign a priority level to the filter pipeline.
        # The priority level determines the order in which the filter pipelines are executed.
        # The lower the number, the higher the priority.
        priority: int = 0

        OPENAI_API_BASE_URL: str = os.getenv("OPENAI_API_BASE_URL")
        OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
        TRANSLATE_MODEL: str = os.getenv("TRANSLATE_MODEL", "gpt-4o-mini")

        # Translate languages
        # Assistant message will be translated from SOURCE_LANGUAGE to TARGET_LANGUAGE
        SOURCE_LANGUAGE: Optional[str] = os.getenv("SOURCE_LANGUAGE", "en")
        TARGET_LANGUAGE: Optional[str] = os.getenv("TARGET_LANGUAGE", "zh-TW")

        # New parameter to control display of both languages
        TRANSLATE_FILTER: bool = os.getenv("TRANSLATE_FILTER", "false").lower() == "true"
        DISPLAY_BOTH_LANGUAGES: bool = os.getenv("DISPLAY_BOTH_LANGUAGES", "true").lower() == "true"

    def __init__(self):
        # Pipeline filters are only compatible with Open WebUI
        # You can think of filter pipeline as a middleware that can be used to edit the form data before it is sent to the OpenAI API.
        self.type = "filter"

        # Optionally, you can set the id and name of the pipeline.
        # Best practice is to not specify the id so that it can be automatically inferred from the filename, so that users can install multiple versions of the same pipeline.
        # The identifier must be unique across all pipelines.
        # The identifier must be an alphanumeric string that can include underscores or hyphens. It cannot contain spaces, special characters, slashes, or backslashes.
        # self.id = "libretranslate_filter_pipeline"
        self.name = "LLM Translate Filter"

        # Initialize
        self.valves = self.Valves()
        if not self.valves.TRANSLATE_FILTER:
            self.valves.pipelines = []

        pass

    async def on_startup(self):
        # This function is called when the server is started.
        print(f"on_startup:{__name__}")
        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        print(f"on_shutdown:{__name__}")
        pass

    async def on_valves_updated(self):
        # This function is called when the valves are updated.
        pass

    def translate(self, text: str, source: str, target: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.valves.OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }

        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": f"Translate the following text to {target}. Provide only the translated text and nothing else.",
                },
                {"role": "user", "content": text},
            ],
            "model": self.valves.TRANSLATE_MODEL,
        }
        print(payload)

        try:
            r = requests.post(
                url=f"{self.valves.OPENAI_API_BASE_URL}/chat/completions",
                json=payload,
                headers=headers,
                stream=True,
            )

            r.raise_for_status()
            response = r.json()
            print(response)
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Error: {e}"

    async def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        print(f"outlet:{__name__}")

        # Check if this is a title response
        if "title" in body:
            print(f"Title response detected: {body['title']}")
            return body

        messages = body.get("messages", [])
        assistant_message = get_last_assistant_message(messages)

        print(f"Original message: {assistant_message}")

        # Translate assistant message
        translated_assistant_message = self.translate(
            assistant_message,
            self.valves.SOURCE_LANGUAGE,
            self.valves.TARGET_LANGUAGE,
        )

        print(f"Translated message: {translated_assistant_message}")

        # Update the last assistant message with the translated content
        for message in reversed(messages):
            if message["role"] == "assistant":
                if self.valves.DISPLAY_BOTH_LANGUAGES:
                    message["content"] = combine_messages(assistant_message, translated_assistant_message)
                else:
                    message["content"] = translated_assistant_message
                break

        body["messages"] = messages
        return body


def combine_messages(original: str, translated: str) -> str:
    def split_message(message: str) -> list:
        parts = re.split(r"\n|(?<=```)", message)
        return [part.strip() for part in parts if part.strip() or part == "```"]

    def format_part(orig: str, trans: str) -> str:
        if orig.startswith("```"):
            return orig
        elif orig[0].isdigit() and orig[1] == "." or orig[0] == "-":
            return f"{orig}\n   {trans.strip()}"
        else:
            return f"{orig}\n{trans}"

    original_parts = split_message(original)
    translated_parts = [p.lstrip("- ").lstrip("0123456789.") for p in split_message(translated)]

    combined = [format_part(orig, trans) for orig, trans in zip(original_parts, translated_parts)]
    result = "\n".join(combined)

    # Add an extra newline after lists end
    result = re.sub(r"(\n   [^\n]+)(\n\d\.|\n-|\n[^\n])", r"\1\n\2", result)

    return result
