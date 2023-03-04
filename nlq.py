import os
from dataclasses import dataclass
from textwrap import dedent

import sqlite3
import openai


@dataclass
class ColumnInfo:
    cid: int
    name: str
    type: str
    notnull: int
    dflt_value: None | int | float| str
    pk: int

    def format(self, desc: str | None = None) -> str:
        null = "NOTNULL" if self.notnull else "NULLABLE"
        descs = ["description:", desc] if desc else []
        return " ".join([self.name, self.type, null] + descs)


@dataclass
class TableInfo:
    columns: list[ColumnInfo]

    def format(self, desc_dict: dict[str, str]) -> str:
        return "\n".join([column.format(desc_dict.get(column.name)) for column in self.columns])

    @classmethod
    def from_rows(
        cls,
        rows: list[tuple[int, str, str, int, None | int | float, int]]
    ) -> "TableInfo":
        columns = [ColumnInfo(*row) for row in rows]
        return TableInfo(columns)


@dataclass
class SQlite:
    conn: sqlite3.Connection
    column_descriptions: dict[str, dict[str, str]] | None = None
    api_key: str | None = None

    def __post_init__(self):
        api_key = self.api_key or os.getenv("OPENAI_API_KEY")
        openai.api_key = api_key

    def get_cursor(self) -> sqlite3.Cursor:
        return self.conn.cursor()

    def table_names(self) -> list[str]:
        q = 'SELECT name FROM sqlite_master WHERE type = "table"'
        return [row[0] for row in self.get_cursor().execute(q)]

    def descriptions(self) -> dict[str, dict[str, str]]:
        return self.column_descriptions or {}

    def format_tables_info(self) -> str:
        q = "PRAGMA table_info({})"
        cursor = self.get_cursor()
        info_list = []
        for table_name in self.table_names():
            info_list.append("")
            info_list.append(f"TABLE_NAME: {table_name}")
            info = TableInfo.from_rows(list(cursor.execute(q.format(table_name))))
            info_list.append(info.format(self.descriptions().get(table_name, {})))
            info_list.append("")
        return "\n".join(info_list).replace("\n", "\n# ")

    def make_prompt(self, query: str) -> str:
        q = query.format("\n", "\n# ")
        return  dedent(
            f"""
            {self.format_tables_info()}
            # {q}
            SELECT *
            """
        )

    def _translate(self, sql: str):
        lines = sql.split("\n")
        lines.append("")
        lines.append("The above SQL can be rewritten as follows in SQLite")
        prompt = "\n".join(["# " + line for line in lines])
        res = openai.Completion.create(
            model="code-davinci-002",
            prompt=prompt + "\n\nSELECT *",
            temperature=0,
            max_tokens=150,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=["#", ";"]
        )
        return "SELECT *\n" + res["choices"][0]["text"]

    def _build_query(
        self,
        query: str,
        *args,
        **kwargs,
    ) -> str:
        res = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0301",
            messages=[
                {"role": "system", "content": "You are helpful assistant that translates Any Language to SQL"},
                {"role": "user", "content": f"{self.make_prompt(query)}"}
            ],
            *args,
            **kwargs,
        )
        return "SELECT *" + res["choices"][0]["message"]["content"]

    def build_query(
        self,
        query: str,
        *args,
        **kwargs,
    ) -> str:
        q = self._build_query(query, *args, **kwargs)
        return self._translate(q)

    def execute(self, query, *args, **kwargs) -> sqlite3.Cursor:
        cursor = self.get_cursor()
        return cursor.execute(self.build_query(query, *args, **kwargs))
