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

### Index a PDF File
```bash
uv run rag.py index ./AI\ Engineer-Lead\ -\ Anexo.pdf
```

### Index Embeddings
```bash
uv run rag.py index-embeddings
```

### Display Help
```bash
uv run rag.py --help
```

### Search Using Full-Text
```bash
uv run rag.py search "Como funciona o programa de recompensas"
```

### Search Using Embeddings
```bash
uv run rag.py search-embeddings "Como funciona o programa de recompensas"
```

### Chat Mode
```bash
uv run rag.py chat "Como funciona o programa de recompensas"
```

## Requirements

- Python 3.8+
- Dependencies as specified in `pyproject.toml`

## License

This project is open source.
