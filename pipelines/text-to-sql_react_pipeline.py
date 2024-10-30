import os
from typing import Generator, Iterator, List, Union

from langchain import hub
from langchain.agents.agent import AgentExecutor
from langchain.agents.react.agent import create_react_agent
from langchain.chains.sql_database.query import create_sql_query_chain
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.prompts.chat import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai.chat_models.base import ChatOpenAI
from pydantic import BaseModel


class Pipeline:
    class Valves(BaseModel):
        OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
        LANGCHAIN_TRACING_V2: bool = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
        LANGCHAIN_API_KEY: str = os.getenv("LANGCHAIN_API_KEY", "")
        LANGCHAIN_PROJECT: str = os.getenv("LANGCHAIN_PROJECT", "")
        pass

    def __init__(self):
        self.name = "Text-to-SQL ReAct"
        self.valves = self.Valves()
        self.chain = self.create_chain()
        pass

    async def on_startup(self):
        print(f"on_startup:{__name__}")
        pass

    async def on_shutdown(self):
        print(f"on_shutdown:{__name__}")
        pass

    def clean_sql(self, sql):
        sql = sql.strip("`").split("\n", 1)[-1].rsplit("\n", 1)[0]
        return sql

    def create_chain(self):
        sql_prompt = hub.pull("langchain-ai/sql-agent-system-prompt")
        react_prompt = hub.pull("hwchase17/react")
        combined_prompt = ChatPromptTemplate.from_messages(
            [
                *sql_prompt.messages,
                ("system", react_prompt.template),
            ]
        )

        db = SQLDatabase.from_uri("sqlite:///databases/Chinook.db")
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        toolkit = SQLDatabaseToolkit(db=db, llm=llm)

        agent = create_react_agent(llm, toolkit.get_tools(), combined_prompt)
        agent_executor = AgentExecutor(agent=agent, tools=toolkit.get_tools())

        return agent_executor

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> Union[str, Generator, Iterator]:
        print(f"pipe:{__name__}")

        try:
            # TODO: IT DOES NOT WORK
            # if body["stream"]:
            #     for step in chain.generate_response(messages):
            #         if "output" in step:
            #             return (chunk for chunk in step["output"])
            # else:
            return self.generate_response(messages)["output"]
        except Exception as e:
            return f"Error: {e}"

    def generate_response(self, input_text: str, stream: bool = False):
        config = {"configurable": {"session_id": "text-to-sql-react-chain-session"}}
        if stream:
            return self.chain.stream({"input": input_text, "dialect": "SQLite", "top_k": 5}, config)
        else:
            return self.chain.invoke({"input": input_text, "dialect": "SQLite", "top_k": 5}, config)

    @staticmethod
    def display_response(response: Union[dict, Generator, Iterator]) -> None:
        if isinstance(response, Generator):
            for chunk in response:
                if "output" in chunk:
                    for c in chunk["output"]:
                        print(c, end="")

        if isinstance(response, dict):
            response = response["output"]

        print(response)
