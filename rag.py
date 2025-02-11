import os

import pymupdf4llm
import typer
from litellm import completion
from pysqlite3 import dbapi2 as sqlite3
from semantic_text_splitter import MarkdownSplitter

DEFAULT_MODEL = os.environ.get(
    "LLM_MODEL", "gemini/gemini-2.0-flash-lite-preview-02-05"
)

RAG_PROMPT = """
You are a helpful assistant.

<Rules>
- Provide clear, concise, and engaging answers
- Use only the provided context to answer the user's question
- If the provided context does not answer the question, say I don't know
- Always add a sources section at the end of the response
- Only include sources that were directly used in crafting your response
</Rules>

<Example>
[detailed response here guided by the Rules and Context]

**Sources:**
- [title]
</Example>
"""

RAG_ITEM = """
<Context>
    <title>{title}</title>
    <content>{content}</content>
</Context>
"""


def create_db():
    db = sqlite3.connect("./db.sqlite")
    db.execute("""
        create table if not exists chunks (
            id integer primary key,
            document varchar,
            chunk_id integer,
            chunk varchar
        );
    """)

    db.execute("""
        create virtual table if not exists fts_chunks using fts5(
          document, chunk,
          content='chunks', content_rowid='id'
        );
    """)

    # Triggers to keep the FTS index up to date.
    db.execute("""
        create trigger if not exists fts_chunks_ai after insert on chunks begin
          insert into fts_chunks(rowid, document, chunk) values (new.id, new.document, new.chunk);
        end;
    """)
    db.execute("""
        create trigger if not exists fts_chunks_ad after delete on chunks begin
          insert into fts_chunks(fts_chunks, rowid, document, chunk) values('delete', old.id, old.document, old.chunk);
        end;
    """)
    db.execute("""
        create trigger if not exists fts_chunks_au after update on chunks begin
          insert into fts_chunks(fts_chunks, rowid, document, chunk) values('delete', old.id, old.document, old.chunk);
          insert into fts_chunks(rowid, document, chunk) values (new.id, new.document, new.chunk);
        end;
    """)

    return db


def search(db, query: str):
    results = db.execute(
        """select rowid, document, chunk, rank
        from fts_chunks where chunk match ? order by rank limit 5""",
        (query,),
    )
    return results.fetchall()


def llm_complete(db, context_search: str, question: str):
    messages = [{"role": "system", "content": RAG_PROMPT}]
    rag_context = search(db, context_search)
    for row in rag_context:
        messages.append(
            {
                "role": "user",
                "content": RAG_ITEM.format(title=row[1], content=row[2]),
            }
        )
    messages.append({"role": "user", "content": question})
    response = completion(model=DEFAULT_MODEL, messages=messages, stream=True)
    for part in response:
        print(part.choices[0].delta.content or "")


def main(action: str, action_object: str):
    db = create_db()

    if action == "index":
        md_text = pymupdf4llm.to_markdown(action_object)
        splitter = MarkdownSplitter(1000)
        chunks = splitter.chunks(md_text)

        for i, chunk in enumerate(chunks):
            db.execute(
                "insert into chunks(document, chunk_id, chunk) values (?, ?, ?)",
                (action_object, i, chunk),
            )
        db.commit()

    elif action == "search":
        top_results = search(db, action_object)
        print(top_results)

    elif action == "chat":
        question = input("Ask me a question: ")
        llm_complete(db, action_object, question)


if __name__ == "__main__":
    typer.run(main)
