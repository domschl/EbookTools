#! /usr/bin/env python3

import logging
import sys
import os
import json
import argparse

from calibre_tools import CalibreTools
from kindle_tools import KindleTools
from md_tools import MdTools
from indra_tools import IndraTools
from metadata import Metadata


if __name__ == "__main__":
    # Init logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Ebook Tools")
    parser.add_argument(
        "-d",
        "--dry-run",
        action="store_true",
        help="Dry run, do not copy or delete files",
    )
    parser.add_argument(
        "-E",
        "--execute",
        action="store_true",
        help="Execute, this can DELETE files, be careful, test first with -d. Note: this option will be removed at some point and be default.",
    )
    parser.add_argument(
        "-x",
        "--delete",
        action="store_true",
        help="Delete files that are debris, DANGER, test first with -d",
    )
    parser.add_argument(
        "-np",
        "--non-interactive",
        action="store_true",
        help="Non-interactive mode, do not show progress bars",
    )
    parser.add_argument(
        "action",
        nargs="*",
        help="Action: export, notes, kindle, indra, meta, timeline",
    )
    parser.add_argument(
        "-t",
        "--time",
        type=str,
        default="",
        help="Timeline time range, can use BP (with option ka, Ma, Ga qualifiers) or BC qualifier, e.g. '10 Ma BP - 1920-03-01' or '1943-04 - 2001-12-31' or '1000 BC - 500', Date format: YYYY[-MM[-DD]] [BC] or year.fraction kya BP or year BP or year ka BP or year Ma BP or year Ga BP. Start and end date are separated by ' - ' with mandatory spaces around the dash, e.g. '10 Ma BP - 1920-03-01'. Default is all.",
    )
    parser.add_argument(
        "-o",
        "--domains",
        type=str,
        default="",
        help="Restrict search domains to list of space separated [Indra-]domains",
    )
    parser.add_argument(
        "-k",
        "--keywords",
        type=str,
        default="",
        help="Restrict search to list of space separated keywords",
    )
    parser.add_argument(
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

    # Set options
    dry_run = args.dry_run
    delete = args.delete
    interactive = not args.non_interactive
    do_export = "export" in args.action
    do_notes = "notes" in args.action
    do_kindle = "kindle" in args.action
    do_indra = "indra" in args.action
    do_meta = "meta" in args.action
    do_timeline = "timeline" in args.action

    if args.execute is False:
        dry_run = True

    if (
        do_export is False
        and do_notes is False
        and do_kindle is False
        and do_indra is False
        and do_meta is False
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
        default_config = {
            "calibre_path": "~/ReferenceLibrary/Calibre Library",
            "kindle_path": "~/Workbench/KindleClippings",
            "meta_path": "~/MetaLibrary",
            "notes_path": "~/Notes",
            "notes_books_subfolder": "Books",
            "export_formats": ["epub", "pdf"],
        }
        # Create dir
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        with open(config_file, "w") as f:
            f.write(json.dumps(default_config, indent=4))
        logger.info(f"Config file {config_file} created, please edit it")
        exit(1)
    else:
        try:
            with open(config_file, "r") as f:
                config = json.load(f)
            calibre_path = os.path.expanduser(config["calibre_path"])
            kindle_path = os.path.expanduser(config["kindle_path"])
            meta_path = os.path.expanduser(config["meta_path"])
            notes_path = os.path.expanduser(config["notes_path"])
            notes_books_path = os.path.join(notes_path, config["notes_books_subfolder"])
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

    if do_export is True or do_notes is True:
        calibre = CalibreTools(calibre_path=calibre_path)
        logger.info(
            f"Calibre Library {calibre.calibre_path}, loading and parsing XML metadata"
        )
        calibre.load_calibre_library_metadata(progress=interactive)
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
            vacuum=args.vacuum,
        )
        logger.info(
            f"Calibre Library {calibre.calibre_path} export: {new_books} new books, {upd_books} updated books, {debris} debris"
        )
    if do_notes is True or do_indra is True:
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
            note = notes.notes[note_name]
            for table in note["tables"]:
                if "metadata" in table and len(table["metadata"].keys()) > 0:
                    metadata_cnt += 1
                table_cnt += 1
        logger.info(
            f"Loaded {len(notes.notes)} notes with {table_cnt} tables, {metadata_cnt} metadata tables"
        )
        if do_indra is True:
            indra = IndraTools()
            event_cnt = 0
            skipped_cnt = 0
            for note_name in notes.notes:
                note = notes.notes[note_name]
                for table in note["tables"]:
                    new_evs, new_skipped = indra.add_events_from_table(
                        table, check_order=True
                    )
                    event_cnt += new_evs
                    skipped_cnt += new_skipped
            logger.info(
                f"Found {len(indra.events)} (added {event_cnt}) Indra events in notes, skipped {skipped_cnt}"
            )
            if do_timeline is True:
                time_par = args.time
                if time_par is not None:
                    if time_par == "":
                        time_par = None
                domains_par = args.domains
                if domains_par is not None:
                    if domains_par == "":
                        domains_par = None
                    else:
                        domains_par = domains_par.split(" ")
                keywords_par = args.keywords
                if keywords_par is not None:
                    if keywords_par == "":
                        keywords_par = None
                    else:
                        keywords_par = keywords_par.split(" ")
                evts = indra.search_events(
                    time=time_par,
                    domains=domains_par,
                    keywords=keywords_par,
                    in_intervall=False,
                    full_overlap=True,
                    partial_overlap=False,
                )
                if time_par is not None:
                    if len(evts) > 0 and time_par is not None:
                        print(" --------- < ----- > ---------")
                        indra.print_events(evts)
                    evts = indra.search_events(
                        time=time_par,
                        domains=domains_par,
                        keywords=keywords_par,
                        in_intervall=False,
                        full_overlap=False,
                        partial_overlap=True,
                    )
                    if len(evts) > 0:
                        print(" --------- <| ----- |> ---------")
                        indra.print_events(evts)
                    evts = indra.search_events(
                        time=time_par,
                        domains=domains_par,
                        keywords=keywords_par,
                        in_intervall=True,
                        full_overlap=False,
                        partial_overlap=False,
                    )
                    if len(evts) > 0:
                        print(" --------- | ----- | ---------")
                        indra.print_events(evts)
                else:
                    indra.print_events(evts)
        if calibre is not None and do_notes is True:
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
        clippings = []
        if kindle.check_for_connected_kindle() is False:
            logger.error("No Kindle connected!")
            # enumerate txt files in kindle_path
            for root, dirs, files in os.walk(kindle_path):
                for file in files:
                    if file.endswith(".txt"):
                        kindle_path = os.path.join(root, file)
                        clippings_text = kindle.get_clippings_text(kindle_path)
                        clippings.append(kindle.parse_clippings(clippings_text))
        else:
            clippings_text = kindle.get_clippings_text()
            clippings = kindle.parse_clippings(clippings_text)
        if clippings is None or len(clippings) == 0:
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
