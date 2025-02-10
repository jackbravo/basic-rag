import pymupdf4llm
import typer
from pysqlite3 import dbapi2 as sqlite3
from semantic_text_splitter import MarkdownSplitter


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
          chunk,
          content='chunks', content_rowid='id'
        );
    """)

    # Triggers to keep the FTS index up to date.
    db.execute("""
        create trigger if not exists fts_chunks_ai after insert on chunks begin
          insert into fts_chunks(rowid, chunk) values (new.id, new.chunk);
        end;
    """)
    db.execute("""
        create trigger if not exists fts_chunks_ad after delete on chunks begin
          insert into fts_chunks(fts_chunks, rowid, chunk) values('delete', old.id, old.chunk);
        end;
    """)
    db.execute("""
        create trigger if not exists fts_chunks_au after update on chunks begin
          insert into fts_chunks(fts_chunks, rowid, chunk) values('delete', old.id, old.chunk);
          insert into fts_chunks(rowid, chunk) values (new.id, new.chunk);
        end;
    """)

    return db


def main(pdf_path: str):
    md_text = pymupdf4llm.to_markdown(pdf_path)
    splitter = MarkdownSplitter(1000)
    chunks = splitter.chunks(md_text)

    db = create_db()

    for i, chunk in enumerate(chunks):
        db.execute(
            "insert into chunks(document, chunk_id, chunk) values (?, ?, ?)",
            (pdf_path, i, chunk),
        )
    db.commit()


if __name__ == "__main__":
    typer.run(main)
