import ast
import os
import re
from typing import Generator, Iterator, List, Union

from langchain.chains.sql_database.query import create_sql_query_chain
from langchain_community.tools.sql_database.tool import (
    QuerySQLCheckerTool,
    QuerySQLDataBaseTool,
)
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.prompts.chat import PromptTemplate
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
        self.name = "Text-to-SQL"
        self.valves = self.Valves()
        self.db = SQLDatabase.from_uri("sqlite:///databases/Chinook.db")
        pass

    async def on_startup(self):
        print(f"on_startup:{__name__}")
        pass

    async def on_shutdown(self):
        print(f"on_shutdown:{__name__}")
        pass

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> Union[str, Generator, Iterator]:
        print(f"pipe:{__name__}")

        def format_query(query: str) -> str:
            """Format SQL query with proper line breaks and indentation."""
            keywords = ["SELECT", "FROM", "JOIN", "WHERE", "GROUP BY", "HAVING", "ORDER BY", "LIMIT", "OFFSET"]
            query = " ".join(query.strip().split())  # Normalize whitespace

            # Convert keywords to uppercase and add line breaks
            for keyword in keywords:
                query = re.sub(rf"\b{keyword}\b", keyword, query, flags=re.IGNORECASE)
                query = query.replace(f" {keyword} ", f"\n{keyword} ")

            # Special handling for JOIN...ON
            query = query.replace(" ON ", "\n    ON ")

            return query

        def data_to_table(query: str, data: str) -> str:
            """Convert SQL query results to a formatted table."""
            # Extract column headers from SELECT clause
            SELECT_clause = re.search(r"SELECT\s+(.*?)\s+FROM", query, re.IGNORECASE).group(1)
            headers = [col.strip() for col in SELECT_clause.split(",")]
            headers = [re.split(r"\s+as\s+", col.strip(), flags=re.IGNORECASE)[-1] for col in headers]

            # Parse data string into list of tuples
            data_list = ast.literal_eval(data) if isinstance(data, str) else []

            return tabulate(data_list, headers, tablefmt="pretty", showindex=True)

        prompt = PromptTemplate.from_template(
            """
You are a helpful assistant that can answer questions with reference to the SQL query result from the database.
Question:
{question}
Query:
{query}
Result:
{result}
Answer:
"""
        )

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        write_query = create_sql_query_chain(llm, self.db)
        execute_query = QuerySQLDataBaseTool(db=self.db)

        def clean_query(sql):
            cleaned_sql = QuerySQLCheckerTool(db=self.db, llm=llm).invoke(sql)
            return cleaned_sql.replace("```sql", "").replace("```", "").strip()

        chain = (
            RunnableParallel(
                query=write_query,
                question=RunnablePassthrough(),
            )
            | RunnablePassthrough.assign(question=lambda x: x["question"]["question"])
            | RunnablePassthrough.assign(query=lambda x: clean_query(x["query"]))
            | RunnablePassthrough.assign(result=lambda x: execute_query.run(x["query"]))
            | RunnablePassthrough.assign(
                prompt=lambda x: prompt.format(
                    question=x["question"],
                    query=x["query"],
                    result=x["result"],
                )
            )
            | RunnablePassthrough.assign(answer=lambda x: llm.invoke(x["prompt"]))
        )

        try:
            result = chain.invoke({"question": user_message})
            return f"""
{result["answer"]}

```sql
{format_query(result["query"])}

{data_to_table(result["query"], result["result"])}
```
"""
        except Exception as e:
            return f"{e}"
