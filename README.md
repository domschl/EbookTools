# EbookTools

EbookTools is a collection of Python-based utilities for managing ebook libraries, extracting metadata, processing highlights, and generating temporal data (timelines) from various sources.

> [!NOTE]
> This project is currently under active development. The documentation focuses on the project structure and component modularity to facilitate reuse and further development.

## Project Structure

The project is organized into several functional components designed to be modular and reusable.

### Core Utilities
- [ebook_utils.py](file:///home/dsc/Codeberg/EbookTools/ebook_utils.py): Common utility functions, including text sanitization for file systems and Unicode-based progress bar generation.
- [metadata.py](file:///home/dsc/Codeberg/EbookTools/metadata.py): A unified interface for extracting metadata from PDF and EPUB files using `pypdf` and `lxml`.

### Platform & Device Integrations
- [calibre_tools.py](file:///home/dsc/Codeberg/EbookTools/calibre_tools.py): Tools for interfacing with Calibre libraries, including loading metadata from `metadata.db` (XML export) and exporting books/metadata to structured formats.
- [kindle_tools.py](file:///home/dsc/Codeberg/EbookTools/kindle_tools.py): Parser for Kindle clippings (`My Clippings.txt`). It handles the complex, locale-specific formatting of highlights and notes.
- [localization](file:///home/dsc/Codeberg/EbookTools/calibre_tools_localization.py): Support files like `calibre_tools_localization.py` and `kindle_tools_localization.py` provide locale-specific string mappings.

### Content & Markdown Processing
- [md_tools.py](file:///home/dsc/Codeberg/EbookTools/md_tools.py): Utilities for working with Markdown files, specifically focused on extracting and generating Markdown tables with embedded YAML metadata.
- [conv.py](file:///home/dsc/Codeberg/EbookTools/conv.py): LaTeX to SVG conversion utility using `matplotlib`, useful for rendering formulas in ebook contexts.

### AI-Powered Search
- [ai_search.py](file:///home/dsc/Codeberg/EbookTools/ai_search.py): Implements document search using sentence embeddings. It leverages `sentence_transformers` and `torch` to index and query document content based on semantic similarity.

### Timelines & Historical Data
- [time_lines.py](file:///home/dsc/Codeberg/EbookTools/time_lines.py): Core logic for processing and filtering temporal events. It integrates with `indralib` for advanced time handling (BP, BC, geological scales).
- [wiki_timelines_exporters/](file:///home/dsc/Codeberg/EbookTools/wiki_timelines_exporters): Scripts and notebooks for scraping and exporting timeline data from Wikipedia.

### Experimental & UI
- [pgl.py](file:///home/dsc/Codeberg/EbookTools/pgl.py): A low-level graphics/UI abstraction layer built on `sdl2` (PySDL2).
- [indra_tools.py](file:///home/dsc/Codeberg/EbookTools/indra_tools.py): Integration points for the Indra data ecosystem.

---

## Component Reuse

Most modules are designed to be imported independently:

- **Metadata Extraction**: Use `Metadata` from `metadata.py` for a simple EPUB/PDF metadata API.
- **Kindle Parsing**: Use `KindleTools` from `kindle_tools.py` to parse clipping files into JSON-like structures.
- **Markdown Tables**: Use `MdTools` from `md_tools.py` to parse or generate Markdown files with rich metadata.
- **Semantic Search**: Use `ai_search.py` as a standalone library for embedding-based local search.

## Configuration

The main entry point (`ebook_tools.py`) uses a configuration file located at `~/.config/EbookTools/ebook_tools.json`. It defines paths for:
- Calibre Library
- Kindle Clippings
- Metadata Library
- Notes (Markdown) repository

## Key Dependencies

The project uses `uv` for dependency and environment management.

- **PDF/EPUB**: `pypdf`, `lxml`, `pymupdf`
- **NLP/AI**: `sentence-transformers`, `torch`, `einops`, `numpy`
- **Data Analysis**: `pandas`, `polars`, `pyarrow`
- **Graphics & UI**: `PySDL2`, `matplotlib`, `pillow`
- **Utilities**: `beautifulsoup4`, `pyyaml`, `rich`, `pycryptodome`
- **Notebooks**: `jupyterlab`

## Setup & Development

Since this is an ongoing project, it is recommended to use [uv](https://github.com/astral-sh/uv) to manage the environment:

```bash
# Install dependencies
uv sync

# Run the main tool
uv run ebook_tools.py --help
```

### Main Entry Point

The CLI orchestrator is `ebook_tools.py`. It supports various actions:
```bash
python ebook_tools.py export|notes|kindle|indra|meta|timeline|bookdates [options]
```

### Action Overview
- `export`: Copies books from Calibre to the local repository.
- `notes`: Syncs Calibre metadata to Markdown notes.
- `kindle`: Parses connected Kindle clippings.
- `meta`: Bulk metadata extraction test.
- `timeline` / `bookdates`: Processes temporal data from notes and Calibre.

---

## Roadmap & Potential Reuses

- **Modular Parsers**: The `KindleTools` and `Metadata` classes can be extracted for use in other ebook-related projects.
- **Indra Integration**: Ongoing work to bridge ebook metadata with the Indra event/temporal database.
- **AI Search**: The `ai_search.py` module provides a foundation for a local RAG (Retrieval-Augmented Generation) system for personal libraries.
