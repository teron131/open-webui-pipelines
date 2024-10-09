import os
from typing import Generator, Iterator, List, Union

from langchain.agents.agent import AgentExecutor
from langchain.agents.tool_calling_agent.base import create_tool_calling_agent
from langchain.chat_models.base import init_chat_model
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.tools import tool
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
            {"id": "gpt-4o", "name": "GPT-4o"},
            {"id": "gpt-4o-mini", "name": "GPT-4o-mini"},
            {"id": "o1-mini", "name": "o1 mini"},
            {"id": self.valves.AZURE_OPENAI_DEPLOYMENT_NAME, "name": "Azure GPT-4o"},
            {"id": "gemini-1.5-pro", "name": "Gemini 1.5 Pro"},
            {"id": "gemini-1.5-flash", "name": "Gemini 1.5 Flash"},
            {"id": "claude-3.5-sonnet", "name": "Claude 3.5 Sonnet"},
        ]

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> Union[str, Generator, Iterator]:
        print(f"pipe:{__name__}")
        print(f"model_id: {model_id}")

        llm = self.get_llm(model_id)
        tools = self.get_tools()
        chain = self.universal_chain(llm, tools)

        print(body)

        try:
            if body["stream"]:
                for step in chain.stream({"input": messages}):
                    if "output" in step:
                        return (chunk for chunk in step["output"])
            else:
                return chain.invoke({"input": messages})["output"]
        except Exception as e:
            return f"Error: {e}"

    def get_llm(self, model_id: str):
        try:
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
                llm = init_chat_model(model=model_id)
        except Exception as e:
            raise ValueError(f"Invalid model_id: {model_id}")
        return llm

    def get_tools(self):
        @tool
        def add(a: float, b: float) -> float:
            """Adds a and b."""
            return a + b

        @tool
        def multiply(a: float, b: float) -> float:
            """Multiplies a and b."""
            return a * b

        return [add, multiply]

    def universal_chain(self, llm, tools):
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful assistant"),
                ("human", "{input}"),
                # Placeholders fill up a **list** of messages
                ("placeholder", "{agent_scratchpad}"),
            ]
        )

        agent = create_tool_calling_agent(llm, tools, prompt)
        return AgentExecutor(agent=agent, tools=tools)
