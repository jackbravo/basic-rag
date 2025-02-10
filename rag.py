import typer
import pymupdf4llm
from semantic_text_splitter import MarkdownSplitter


def main(pdf_path: str):
    md_text = pymupdf4llm.to_markdown(pdf_path)
    splitter = MarkdownSplitter(1000)
    chunks = splitter.chunks(md_text)
    print(chunks)


if __name__ == "__main__":
    typer.run(main)
