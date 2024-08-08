import logging
import os
import sys
import json

from calibre_tools import CalibreTools
from kindle_tools import KindleTools
from md_tools import MdTools
from indra_tools import IndraTools


if __name__ == "__main__":
    # Init logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    dry_run = True
    delete = False
    interactive = True
    do_export = False
    do_notes = False
    do_kindle = False
    do_indra = False

    args = sys.argv[1:]
    if "-E" in args:
        dry_run = False
        args.remove("-E")
    if "-d" in args:
        dry_run = True
        args.remove("-d")
    if "-np" in args:
        interactive = False
        args.remove("-np")
    if "-x" in args:
        delete = True
        args.remove("-x")
    if "-h" in args:
        print("Usage: python ebook_tools.py [export] [notes] [kindle] [-d] [-x]")
        print("  export: export books from Calibre Library to MetaLibrary folder")
        print("  notes:  export metadata to Notes as Markdown files")
        print("  kindle: export Kindle clippings to Notes as Markdown files")
        print(
            "  indra: search notes for Indra events from tables with first column 'Date'"
        )
        print("  -d: dry run, do not copy or delete files")
        print("  -E: execute, this can DELETE files, be careful, test first with -d")
        print("  -x: delete files that are debris, DANGER, test first with -d")
        print("  -np: non-interactive mode, do not show progress bars")
        exit(0)
    if "export" in args:
        args.remove("export")
        do_export = True
    if "notes" in args:
        args.remove("notes")
        do_notes = True
    if "indra" in args:
        args.remove("indra")
        do_indra = True
    if "kindle" in args:
        args.remove("kindle")
        do_kindle = True
    if (
        do_export is False
        and do_notes is False
        and do_kindle is False
        and do_indra is False
    ):
        logger.error("No action specified, exiting, use -h for help")
        exit(1)

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
    if do_export is True:
        logger.info(f"Calibre Library {calibre.calibre_path}, copying books")
        new_books, upd_books, debris = calibre.export_calibre_books(
            meta_path,
            format=export_formats,
            dry_run=dry_run,
            delete=delete,
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
        )
        table_cnt = 0
        metadata_cnt = 0
        for note_name in notes.notes:
            note = notes.notes[note_name]
            table_cnt += len(note["tables"])
        logger.info(
            f"Loaded {len(notes.notes)} notes with {table_cnt} tables, {metadata_cnt} metadata tables"
        )
        if do_indra is True:
            indra = IndraTools()
            event_cnt = 0
            for note_name in notes.notes:
                note = notes.notes[note_name]
                for table in note["tables"]:
                    new_evs = indra.add_events_from_table(table, note)
                    event_cnt += new_evs
            logger.info(
                f"Found {len(indra.events)} (added {event_cnt}) Indra events in notes"
            )
            indra.print_event()
        if do_notes is True:
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
