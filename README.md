# RAG Script

This project provides a Retrieval Augmented Generation (RAG) CLI application built with Python.

## Overview

The script `rag.py` implements a RAG workflow which:
- Indexes PDF documents converted to Markdown using `pymupdf4llm`
- Splits content into chunks with `semantic_text_splitter`
- Indexes embeddings using `SentenceTransformer`
- Supports full text search and embeddings-based search of indexed chunks
- Offers a chat interface that leverages an LLM (default model: gemini/gemini-2.0-flash-lite-preview-02-05) for question answering

## Usage

## Installation

Install [uv](https://docs.astral.sh/uv/) package manager and run:

```bash
uv venv # to create a virtual environment
uv sync # to install (sync) all dependencies
```

### Index a PDF File

This will also create a FULL TEXT index for the document using fts5 sqlite extension.

```bash
uv run rag.py index ./AI\ Engineer-Lead\ -\ Anexo.pdf
```

### Index Embeddings

This will create an EMBEDDINGS index for the document, in a separate table.
This needs to be run after `index`.
It deletes all data on the embeddings table before indexing again.

We are using `jinaai/jina-embeddings-v3` as embedding model, which is multilingual, it has a cc-nc (non-comercial) license.

```bash
uv run rag.py index-embeddings
```

### Search Using Full-Text

Words are separated by OR when sending the `match` query to SQLite, so the search is for any of the words in the query.

```bash
uv run rag.py search "A question using full text search"
```

### Search Using Embeddings

```bash
uv run rag.py search-embeddings "The top results of a question using embeddings"
```

### Chat Mode

This will use the top results from the search to answer questions using an LLM.
You need to set the `LLM_MODEL` environment variable to something like `gemini/gemini-2.0-flash-lite-preview-02-05`
and the appropriate key using [litellm](https://github.com/BerriAI/litellm), like `GEMINI_API_KEY`.

```bash
uv run rag.py chat "Como funciona o programa de recompensas"
```

## Requirements

- Python 3.8+
- Dependencies as specified in `pyproject.toml`

## Tests

To run tests, execute the following command:

```bash
uv run test_rag_compare.py
```

## License

This project is open source.
