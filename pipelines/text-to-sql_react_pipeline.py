import os
import re
from typing import Generator, Iterator, List, Union

from langchain import hub
from langchain.agents.agent import AgentExecutor
from langchain.agents.chat.output_parser import ChatOutputParser
from langchain.agents.react.agent import create_react_agent
from langchain.chains.sql_database.query import create_sql_query_chain
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_community.utilities import SQLDatabase
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.prompts.chat import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai.chat_models.base import ChatOpenAI
from pydantic import BaseModel
from tabulate import tabulate


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
        agent_executor = AgentExecutor(agent=agent, tools=toolkit.get_tools(), return_intermediate_steps=True)

        return agent_executor

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> Union[str, Generator, Iterator]:
        print(f"pipe:{__name__}")

        def format_sql_query(sql: str) -> str:
            # Keywords to add newlines before
            keywords = ["SELECT", "FROM", "JOIN", "WHERE", "GROUP BY", "HAVING", "ORDER BY", "LIMIT", "OFFSET"]
            sql = " ".join(sql.strip().split())  # Remove extra whitespace and normalize spacing
            for keyword in keywords:
                sql = re.sub(rf"\b{keyword}\b", keyword, sql, flags=re.IGNORECASE)  # Convert keywords to uppercase for case insensitivity
            sql = sql.replace(" ON ", "\n    ON ")  # Handle JOIN ... ON as a special case
            for keyword in keywords:
                sql = sql.replace(f" {keyword} ", f"\n{keyword} ")  # Add newlines before keywords
            return sql

        def data_to_table(query: str, data: str) -> str:
            # Extract column names from SELECT statement
            select_pattern = r"SELECT\s+(.*?)\s+FROM"
            select_clause = re.search(select_pattern, query, re.IGNORECASE).group(1)
            headers = [col.strip().split(" AS ")[-1].strip() for col in select_clause.split(",")]

            # Format data
            if isinstance(data, str):
                data_list = data.strip("[]").split("),")
                data_list = [tuple(item.strip(" ()'").split("',")) for item in data_list if item.strip()]

            return tabulate(data_list, headers=headers, tablefmt="pretty")

        config = {"configurable": {"session_id": "text-to-sql-react-chain-session"}}
        dialect = "SQLite"
        top_k = 20

        try:
            response = self.chain.invoke({"input": user_message, "dialect": dialect, "top_k": top_k}, config)
            last_step = response["intermediate_steps"][-1]
            if last_step[0].tool == "sql_db_query":
                query = last_step[0].tool_input
                data = last_step[1]
            return f"{response["output"]}\n```sql\n{query}\n\n{data_to_table(query, data)}\n```"

        except Exception as e:
            return f"Error: {e}"

    def generate_response(self, input_text: str, stream: bool = False):
        config = {"configurable": {"session_id": "text-to-sql-react-chain-session"}}
        if stream:
            return self.chain.stream({"input": input_text, "dialect": "SQLite", "top_k": 20}, config)
        else:
            return self.chain.invoke({"input": input_text, "dialect": "SQLite", "top_k": 20}, config)

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
