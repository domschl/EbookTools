#! /usr/bin/env python3

import logging
import sys
import os
import json
import argparse
from argparse import ArgumentParser
from typing import TypedDict, cast, Any

from calibre_tools import CalibreTools
from kindle_tools import KindleTools
from md_tools import MdTools, MDTable
from indra_tools import IndraTools
from metadata import Metadata
from time_lines import TimeLines
from ai_search import EmbeddingSearch


class ConfigDict(TypedDict):
     calibre_path: str
     kindle_path: str
     meta_path: str
     book_text_lib: str
     notes_path: str
     notes_books_subfolder: str
     book_text_lib_embeddings: str
     embeddings_model: str
     chunk_size: int
     chunk_overlap: int
     export_formats: list[str]
     

if __name__ == "__main__":
    # Init logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Parse command line arguments
    parser: ArgumentParser = argparse.ArgumentParser(description="Ebook Tools")
    _ = parser.add_argument(
        "-d",
        "--dry-run",
        action="store_true",
        help="Dry run, do not copy or delete files",
    )
    _ = parser.add_argument(
        "-E",
        "--execute",
        action="store_true",
        help="Execute, this can DELETE files, be careful, test first with -d. Note: this option will be removed at some point and be default.",
    )
    _ = parser.add_argument(
        "-x",
        "--delete",
        action="store_true",
        help="Delete files that are debris, DANGER, test first with -d",
    )
    _ = parser.add_argument(
        "-n",
        "--non-interactive",
        action="store_true",
        help="Non-interactive mode, do not show progress bars",
    )
    _ = parser.add_argument(
        "action",
        nargs="*",
        default="",
        help="Action: export, notes, kindle, indra, meta, timeline, bookdates, embed, search",
    )
    _ = parser.add_argument(
        "-t",
        "--time",
        type=str,
        default="",
        help="Timeline time range, can use BP (with option ka, Ma, Ga qualifiers) or BC qualifier," +\
             " e.g. '10 Ma BP - 1920-03-01' or '1943-04 - 2001-12-31' or '1000 BC - 500', Date format:" +\
             " YYYY[-MM[-DD]] [BC] or year.fraction kya BP or year BP or year ka BP or year Ma BP or year Ga BP." +\
             " Start and end date are separated by ' - ' with mandatory spaces around the dash," +\
             " e.g. '10 Ma BP - 1920-03-01'. Default is all.",
    )
    _ = parser.add_argument(
        "-o",
        "--domains",
        type=str,
        default="",
        help="Restrict search domains to list of space separated [Indra-]domains, leading '!' used for exclusion (negation)",
    )
    _ = parser.add_argument(
        "-k",
        "--keywords",
        type=str,
        default="",
        help="Restrict search to list of space separated keywords, leading '!' used for exclusion (negation)," +\
        " '*' for wildcards at beginning, middle or end of keywords." +\
        " Multiple space separated keywords are combined with AND, use '|' for OR combinations.",
    )
    _ = parser.add_argument(
        "-f",
        "--format",
        type=str,
        default="ascii",
        help="Format for timeline table output: none (markdown) or ascii (default)",
    )
    _ = parser.add_argument(
        "-S",
        '--SHA256',
        action="store_true",
        help="Use SHA256 for metadata comparison, default is CRC32",
    )
    _ = parser.add_argument(
        "-V", "--vacuum", action="store_true", help="Show possible debris"
    )
    # Add max_notes, number of notes processed, default=0 which is all:
    # parser.add_argument(
    #     "-m",
    #     "--max-notes",
    #     type=int,
    #     default=0,
    #     help="Max number of notes to process, default=0 which is all",
    # )
    args = parser.parse_args()

    action = cast(str, args.action)
    # Set options
    dry_run: bool = cast(bool, args.dry_run)
    delete: bool = cast(bool, args.delete)
    interactive: bool = not cast(bool, args.non_interactive)
    do_export: bool = "export" in action
    do_notes: bool = "notes" in action
    do_kindle: bool = "kindle" in action
    do_indra: bool = "indra" in action
    do_meta: bool = "meta" in action
    do_timeline: bool = "timeline" in action
    do_bookdates: bool = "bookdates" in action
    do_embed: bool = "embed" in action
    do_search: bool = "search" in action
    do_date_stuff: bool = (do_bookdates is True or do_timeline is True)
    use_sha256 = cast(bool, args.SHA256)

    emb = None
    if cast(bool, args.execute) is False:
        dry_run = True

    if (
        do_export is False
        and do_notes is False
        and do_kindle is False
        and do_indra is False
        and do_meta is False
        and do_bookdates is False
        and do_embed is False
        and do_search is False
    ):
        logger.error("No action specified, exiting, use -h for help")
        exit(1)

    if sys.platform == "haiku1":
        config_file = os.path.expanduser(
            "~/config/settings/EbookTools/ebook_tools.json"
        )
    else:
        config_file = os.path.expanduser("~/.config/EbookTools/ebook_tools.json")

    if os.path.exists(config_file) is False:
        default_config: ConfigDict = {
            "calibre_path": "~/ReferenceLibrary/Calibre Library",
            "kindle_path": "~/Workbench/KindleClippings",
            "meta_path": "~/MetaLibrary",
            "book_text_lib": "~/BookTextLibrary",
            "notes_path": "~/Notes",
            "notes_books_subfolder": "Books",
            "book_text_lib_embeddings": "~/BookTextLibraryEmbeddings",
             "embeddings_model": "nomic-embed-text",
             "chunk_size": 2048,
             "chunk_overlap": 0,
            "export_formats": ["epub", "pdf"],
        }
        # Create dir
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        with open(config_file, "w") as f:
            _ = f.write(json.dumps(default_config, indent=4))
        logger.info(f"Config file {config_file} created, please edit it")
        exit(1)
    else:
        try:
            with open(config_file, "r") as f:
                config: ConfigDict = json.load(f)
            calibre_path: str = os.path.expanduser(config["calibre_path"])
            kindle_path = os.path.expanduser(config["kindle_path"])
            meta_path = os.path.expanduser(config["meta_path"])
            book_text_lib = os.path.expanduser(config["book_text_lib"])
            notes_path = os.path.expanduser(config["notes_path"])
            notes_books_path = os.path.join(notes_path, config["notes_books_subfolder"])
            book_text_lib_embeddings: str = os.path.expanduser(config["book_text_lib_embeddings"])
            embeddings_model = config["embeddings_model"]
            chunk_size = config["chunk_size"]
            chunk_overlap = config["chunk_overlap"]
            export_formats = config["export_formats"]
        except Exception as e:
            logger.error(f"Error reading config file {config_file}: {e}")
            exit(1)
    paths = [calibre_path, kindle_path, meta_path, notes_path]
    for p in paths:
        if os.path.exists(p) is False:
            logger.error(
                f"Path {p} does not exist, please check config file {config_file} or create the directly"
            )
            exit(1)

    if do_date_stuff is True:
        timelines = TimeLines()
    else:
        timelines = None

    if do_export is True or do_notes is True or do_date_stuff is True:
        calibre = CalibreTools(calibre_path=calibre_path)
        logger.info(
            f"Calibre Library {calibre.calibre_path}, loading and parsing XML metadata"
        )
        lastest_mod_time = calibre.load_calibre_library_metadata(progress=interactive, use_sha256=use_sha256, load_text=do_date_stuff)
        logger.info("Calibre Library loaded")
    else:
        calibre = None

    if calibre is not None and do_export is True:
        logger.info(f"Calibre Library {calibre.calibre_path}, copying books")
        new_books, upd_books, debris = calibre.export_calibre_books(
            meta_path,
            format=export_formats,
            dry_run=dry_run,
            delete=delete,
            vacuum= cast(bool, args.vacuum),
        )
        logger.info(
            f"Calibre Library {calibre.calibre_path} export: {new_books} new books, {upd_books} updated books, {debris} debris"
        )
        if book_text_lib != "" and os.path.exists(book_text_lib):
            logger.info(f"Calibre Library {calibre.calibre_path}, copying text books")
            txt_books, upd_txt_books, txt_debris = calibre.export_calibre_books(
                book_text_lib,
                format=["txt"],
                dry_run=dry_run,
                delete=delete,
                vacuum=cast(bool, args.vacuum),
            )
            logger.info(
                f"Calibre Library {calibre.calibre_path} export: {txt_books} new books, {upd_txt_books} updated books, {txt_debris} debris"
            )
            

    if do_notes is True or do_indra is True or do_date_stuff is True:
        logger.info(f"Loading notes from {notes_path}")
        notes = MdTools(
            notes_folder=notes_path,
            notes_books_folder=notes_books_path,
            progress=interactive,
            dry_run=dry_run,
        )
        table_cnt = 0
        metadata_cnt = 0
        for note_name in notes.notes:
            note: dict[str, Any] = notes.notes[note_name]  # pyright: ignore[reportExplicitAny]
            tables: list[MDTable] = note["tables"]
            for table in tables:
                if "metadata" in table and len(table["metadata"].keys()) > 0:
                    metadata_cnt += 1
                table_cnt += 1
        logger.info(
            f"Loaded {len(notes.notes)} notes with {table_cnt} tables, {metadata_cnt} metadata tables"
        )
    else:
        notes = None

    if do_indra is True:
        if do_indra is True:
            indra = IndraTools()

    if do_date_stuff is True:
        if calibre is not None and timelines is not None:
            _ = timelines.add_book_events(calibre.lib_entries)
        if notes is not None and timelines is not None:
            timelines.add_notes_events(notes)
            format_spec = cast(str, args.format).lower()
            time_spec = cast(str, args.time)
            domains_spec = cast(str, args.domains)
            keywords_spec = cast(str, args.keywords)
            print(f"Format: {format_spec}, time: {time_spec}")
            timelines.notes_rest(do_timeline, format_spec, time_spec, domains_spec, keywords_spec)

    if calibre is not None and do_notes is True and notes is not None:
        logger.info(f"Exporting metadata to {notes_books_path}")
        n, errs, content_updates = calibre.export_calibre_metadata_to_markdown(
            notes,
            notes_books_path,
            dry_run=dry_run,
            delete=delete,
        )
        logger.info(
            f"Exported {n} books to {notes_books_path}, content of {content_updates} books updated"
        )

    if do_kindle is True:
        kindle = KindleTools()
        clippings: list[dict[str, Any]] = []  # pyright: ignore[reportExplicitAny]
        if kindle.check_for_connected_kindle() is False:
            logger.error("No Kindle connected!")
            # enumerate txt files in kindle_path
            for root, dirs, files in os.walk(kindle_path):
                for file in files:
                    if file.endswith(".txt"):
                        kindle_path = os.path.join(root, file)
                        clippings_text = kindle.get_clippings_text(kindle_path)
                        if clippings_text is not None:
                             clips = kindle.parse_clippings(clippings_text)
                             if clips is not None:
                                  clippings += clips
        else:
            clippings_text = kindle.get_clippings_text()
            if clippings_text is not None:
                 clips = kindle.parse_clippings(clippings_text)
                 if clips is not None:
                      clippings += clips
        if len(clippings) == 0:
            logger.error("No clippings found, something went wrong...")
            exit(1)

        logger.info(f"Found {len(clippings)} clippings")
        for i in range(0, 10):
            print(clippings[i])
    if do_meta is True:
        # enumerate files in meta_path
        m_errs = 0
        m_oks = 0
        for root, dirs, files in os.walk(meta_path):
            for file in files:
                if file.endswith(".pdf") or file.endswith(".epub"):
                    file_path = os.path.join(root, file)
                    # print("------------")
                    md = Metadata(file_path)
                    # print(file_path)
                    meta = md.get_metadata()
                    if meta is None:
                        m_errs += 1
                    else:
                        m_oks += 1
                        # print(meta)
        logger.info(f"Processed metadata, ok={m_oks}, errors={m_errs}")
    if do_embed is True:
        if emb is None:
             emb = EmbeddingSearch(embeddings_path = book_text_lib_embeddings)
        logger.info(f"Loading text library: {book_text_lib}")
        book_cnt = emb.read_text_library("CalText", book_text_lib, extensions=[".txt"])
        book_cnt += emb.read_text_library(library_name="CalNotes", library_path=notes_path, extensions=[".md"])
        logger.info(f"{book_cnt} text books+notes loaded, loading PDFs...")
        book_cnt += emb.read_pdf_library(library_name="CalPdf", library_path=meta_path)
        logger.info(f"{book_cnt} pdf+text books loaded, generating embeddings...")
        emb.gen_embeddings(model=embeddings_model, library_name="CalText", verbose=True)
        logger.info("Text embeddings processed")
        emb.gen_embeddings(model=embeddings_model, library_name="CalPdf", verbose=True)
        logger.info("PDF embeddings processed")
    if do_search is True:
        search_spec = cast(str, args.keywords)
        if search_spec == "":
             logger.error("Please specify a search-string with `-k` parameter")
             exit(1)
        if emb is None:
             logger.info("Loading embeddings...")
             emb = EmbeddingSearch(embeddings_path = book_text_lib_embeddings)
             logger.info("Embeddings loaded, searching...")
        best_doc, best_index, best_chunk, cos_val, yellow_liner = emb.search_embeddings(model=embeddings_model, search_text=search_spec, yellow_liner=True, context=20)
        y_min: float | None = None
        y_max: float | None = None
        if yellow_liner is not None:
            for y in yellow_liner:
                 if y_min is None or y<y_min:
                      y_min = y
                 if y_max is None or y>y_max:
                      y_max = y
        if y_min == None:
             y_min = 0
        if y_max == None:
             y_max = 1
        print("-----------------------------------------------")
        print(f"Document: {best_doc}[{best_index}], certainty: {cos_val * 100.0:2.1f} %")
        print("-----------------------------------------------")
        # print(best_chunk)
        print(y_min, y_max)
        if yellow_liner is not None:
            from rich.console import Console
            console = Console()
            line = ""
            for i, c in enumerate(best_chunk):
                 yel = (yellow_liner[i]-y_min)/(y_max - y_min)
                 if yel < 0.5:
                      yel = 0.0
                 col = hex(255 - int(yel*127.0))[2:]
                 line += f"[black on #FFFF{col}]"+c+"[/]"
            # print(line)
            console.print(line)
        print("-----------------------------------------------")
        
    if do_bookdates is True:
        logger.error(f"Can't access the book library texts at {book_text_lib}")
