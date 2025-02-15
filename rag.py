import os

import pymupdf4llm
import typer
from litellm import completion
from prettytable import PrettyTable
from pysqlite3 import dbapi2 as sqlite3
from rerankers import Document, Reranker
from rerankers.results import Result
from semantic_text_splitter import MarkdownSplitter
from sentence_transformers import SentenceTransformer
from sqlite_vec import load, serialize_float32
from tqdm import tqdm
from typing_extensions import Annotated

LIMIT = 5
DEFAULT_MODEL = os.environ.get(
    "LLM_MODEL", "gemini/gemini-2.0-flash-lite-preview-02-05"
)
DEFAULT_RERANKER = os.environ.get(
    "LLM_RERANKER", "corrius/cross-encoder-mmarco-mMiniLMv2-L12-H384-v1"
)
DEFAULT_RER_MODEL_TYPE = os.environ.get("RERANKER_MODEL_TYPE", "cross-encoder")
_cached_model = None
_cached_reranker = None

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
    db.enable_load_extension(True)
    load(db)
    db.enable_load_extension(False)
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

    db.execute("""
        create table if not exists chunks_embeddings (
            id integer primary key,
            embedding float[1024]
                check(
                    typeof(embedding) == 'blob'
                    and vec_length(embedding) == 1024
                )
        )
    """)

    return db


def get_model():
    global _cached_model
    if _cached_model is None:
        print("Initializing model...")
        _cached_model = SentenceTransformer(
            "jinaai/jina-embeddings-v3", trust_remote_code=True
        )
    return _cached_model


def get_reranker():
    global _cached_reranker
    if _cached_reranker is None:
        print("Initializing reranker...")
        print(DEFAULT_RER_MODEL_TYPE)
        _cached_reranker = Reranker(DEFAULT_RERANKER, model_type=DEFAULT_RER_MODEL_TYPE)
    return _cached_reranker


def embeddings_index(db):
    print("Deleting existing embeddings...")
    db.execute("DELETE FROM chunks_embeddings")

    print("Initializing model...")
    model = get_model()

    results = db.execute("SELECT id, chunk FROM chunks").fetchall()
    for id, chunk in tqdm(results, desc="Indexing embeddings"):
        task = "retrieval.passage"
        embeddings = model.encode(chunk, task=task, prompt_name=task).tolist()
        db.execute(
            "INSERT INTO chunks_embeddings(id, embedding) VALUES (?, ?)",
            (
                id,
                serialize_float32(embeddings),
            ),
        )
    db.commit()


def search(db, query: str, limit: int = LIMIT):
    # make words separated by ' OR ' to search for any of them
    # e.g. "hello world" -> "hello OR world"
    query = " OR ".join(query.split())
    results = db.execute(
        f"""select rowid, document, chunk, rank
        from fts_chunks where chunk match ? order by rank limit {limit}""",
        (query,),
    )
    return results.fetchall()


def search_embeddings(db, query: str, limit: int = LIMIT):
    model = get_model()
    task = "retrieval.query"
    embeddings = model.encode(query, task=task, prompt_name=task).tolist()

    sql = f"""
        SELECT c.id, c.document, c.chunk, vec_distance_L2(e.embedding, ?) as distance
        FROM chunks_embeddings e
        LEFT JOIN chunks c USING (id)
        ORDER BY distance
        LIMIT {limit}
    """
    return db.execute(sql, (serialize_float32(embeddings),)).fetchall()


def search_rerank(db, emb_results, query: str, limit: int = LIMIT):
    docs = [
        Document(
            text=doc[2],
            metadata={"title": doc[1], "id": doc[0]},
        )
        for doc in emb_results
    ]
    reranker = get_reranker()
    # error out if ranker is None
    assert reranker is not None, "Ranker not found"
    results = reranker.rank(query, docs)
    top = results.top_k(limit)
    return [
        (doc.metadata["id"], doc.metadata["title"], doc.text, doc.score) for doc in top
    ]


def llm_complete(db, question: str):
    messages = [{"role": "system", "content": RAG_PROMPT}]
    rag_context = search(db, question)
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


def print_results(results):
    table = PrettyTable()
    table.field_names = ["id", "document", "rank"]
    table.align = "l"
    for row in results:
        table.add_row([row[0], row[1], row[3]])
    print(table)


def main(action: str, action_object: Annotated[str | None, typer.Argument()] = None):
    db = create_db()

    if action == "markdown":
        print(pymupdf4llm.to_markdown(action_object))

    elif action == "index":
        md_text = pymupdf4llm.to_markdown(action_object)
        splitter = MarkdownSplitter(1000)
        chunks = splitter.chunks(md_text)

        for i, chunk in enumerate(chunks):
            db.execute(
                "insert into chunks(document, chunk_id, chunk) values (?, ?, ?)",
                (action_object, i, chunk),
            )
        db.commit()

    elif action == "index-embeddings":
        embeddings_index(db)

    elif action == "search":
        if action_object is None:
            raise ValueError("Please provide a query to search for.")
        top_results = search(db, action_object)
        print_results(top_results)

    elif action == "search-embeddings":
        if action_object is None:
            raise ValueError("Please provide a query to search for.")
        top_results = search_embeddings(db, action_object)
        print_results(top_results)

    elif action == "search-rerank":
        if action_object is None:
            raise ValueError("Please provide a query to search for.")
        emb_results = search_embeddings(db, action_object, LIMIT * 2)
        top_results = search_rerank(db, emb_results, action_object)
        print_results(top_results)

    elif action == "chat":
        if action_object is None:
            raise ValueError("Please add a question.")
        llm_complete(db, action_object)


if __name__ == "__main__":
    typer.run(main)
