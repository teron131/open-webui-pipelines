import os
from typing import Generator, Iterator, List, Union

from langchain.schema import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_playground.UniversalChain import get_llm, get_tools
from langgraph.prebuilt import create_react_agent
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
            {"id": "chatgpt-4o-latest", "name": "ChatGPT-4o"},
            {"id": "gpt-4o-2024-11-20", "name": "GPT-4o"},
            {"id": "gpt-4o-mini", "name": "GPT-4o-mini"},
            {"id": "o1", "name": "o1"},
            {"id": "o1-mini", "name": "o1 mini"},
            {"id": self.valves.AZURE_OPENAI_DEPLOYMENT_NAME, "name": "Azure GPT-4o"},
            {"id": "gemini-1.5-pro", "name": "Gemini 1.5 Pro"},
            {"id": "gemini-2.0-flash-exp", "name": "Gemini 2.0 Flash"},
            {"id": "claude-3.5-sonnet", "name": "Claude 3.5 Sonnet"},
            {"id": "deepseek-chat", "name": "DeepSeek Chat"},
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
            # Create LLM and tools
            llm = get_llm(model_id)
            tools = get_tools()

            # From UI's messages list to prompt for state_modifier
            lc_messages = [HumanMessage(content=msg["content"]) if msg["role"] == "user" else AIMessage(content=msg["content"]) for msg in messages if msg["role"] in ("user", "assistant")]
            history_prompt = ChatPromptTemplate.from_messages(lc_messages)

            # Create agent with history
            agent = create_react_agent(
                llm,
                tools,
                state_modifier=history_prompt,
            )

            # Invoke agent
            config = {"configurable": {"thread_id": "universal-chain-session"}}
            response = agent.invoke(
                {"messages": [("user", user_message)]},
                config,
            )
            return response["messages"][-1].content

        except Exception as e:
            return f"Error: {e}"
