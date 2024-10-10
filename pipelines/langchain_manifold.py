import os
from typing import Generator, Iterator, List, Union

import opencc
from dotenv import load_dotenv
from langchain import hub
from langchain.agents.agent import AgentExecutor
from langchain.agents.tool_calling_agent.base import create_tool_calling_agent
from langchain.chat_models.base import init_chat_model
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
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

        chain = UniversalChain(model_name=model_id, use_history=True)

        print(body)

        try:
            # TODO: IT DOES NOT WORK
            # if body["stream"]:
            #     for step in chain.generate_response(messages):
            #         if "output" in step:
            #             return (chunk for chunk in step["output"])
            # else:
            return chain.generate_response(messages)["output"]
        except Exception as e:
            return f"Error: {e}"


class UniversalChain:
    def __init__(self, model_name: str, use_history: bool = False):
        self.llm = self.get_llm(model_name)
        self.tools = self.get_tools()
        self.use_history = use_history
        self.history = InMemoryChatMessageHistory(session_id="universal-chain-session")
        self.chain = self.create_chain()

    def get_llm(self, model_id: str):
        try:
            if "azure" in model_id:
                llm = AzureChatOpenAI(
                    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                )
            elif "gemini" in model_id:
                llm = ChatGoogleGenerativeAI(model=model_id, api_key=os.getenv("GEMINI_API_KEY"))
            elif "claude" in model_id:
                llm = ChatOpenAI(
                    model=f"anthropic/{model_id}",  # Avoid making model_id with '/', otherwise it will mess up the FastAPI URL
                    base_url="https://openrouter.ai/api/v1",
                    api_key=os.getenv("OPENROUTER_API_KEY"),
                )
            elif "gpt" in model_id:
                llm = ChatOpenAI(model=model_id, api_key=os.getenv("OPENAI_API_KEY"))
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

    def create_chain(self):
        tool_agent_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful assistant"),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )
        agent = create_tool_calling_agent(self.llm, self.tools, tool_agent_prompt)
        agent_executor = AgentExecutor(agent=agent, tools=self.tools)

        if self.use_history:
            return RunnableWithMessageHistory(
                agent_executor,
                # This is needed because in most real world scenarios, a session id is needed
                # It isn't really used here because we are using a simple in memory ChatMessageHistory
                lambda session_id: self.history,
                input_messages_key="input",
                history_messages_key="chat_history",
            )
        return agent_executor

    def generate_response(self, input_text: str, stream: bool = False):
        config = {"configurable": {"session_id": "universal-chain-session"}}
        if stream:
            return self.chain.stream({"input": input_text}, config)
        else:
            return self.chain.invoke({"input": input_text}, config)

    @staticmethod
    def s2hk(content: str) -> str:
        converter = opencc.OpenCC("s2hk")
        return converter.convert(content)

    @staticmethod
    def display_response(response: Union[dict, Generator, Iterator]) -> None:
        if isinstance(response, Generator):
            for chunk in response:
                if "output" in chunk:
                    for c in chunk["output"]:
                        print(c, end="")
        elif isinstance(response, dict):
            response = response["output"]
            print(response)
