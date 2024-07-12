import logging
import os
import sys
import json

from calibre_tools import CalibreTools
from kindle_tools import KindleTools

if __name__ == "__main__":
    # Init logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    dry_run = True
    delete = False
    do_export = False
    do_notes = False
    do_kindle = False

    args = sys.argv[1:]
    if "-d" in args:
        dry_run = True
        args.remove("-d")
    if "-E" in args:
        dry_run = False
        args.remove("-E")
    if "-x" in args:
        delete = True
        args.remove("-x")
    if "-h" in args:
        print("Usage: python ebook_tools.py [export] [notes] [kindle] [-d] [-x]")
        print("  export: export books from Calibre Library to MetaLibrary folder")
        print("  notes:  export metadata to Notes as Markdown files")
        print("  kindle: export Kindle clippings to Notes as Markdown files")
        print("  -d: dry run, do not copy files")
        print("  -E: execute, this can DELETE files, be careful, test first with -d")
        print("  -x: delete files that are debris, DANGER, test first with -d")
        exit(0)
    if "export" in args:
        args.remove("export")
        do_export = True
    if "notes" in args:
        args.remove("notes")
        do_notes = True
    if "kindle" in args:
        args.remove("kindle")
        do_kindle = True
    if do_export is False and do_notes is False and do_kindle is False:
        logger.error("No action specified, exiting, use -h for help")
        exit(1)

    config_file = os.path.expanduser("~/.config/EbookTools/ebook_tools.json")

    if os.path.exists(config_file) is False:
        default_config = {
            "calibre_path": "~/ReferenceLibrary/Calibre Library",
            "kindle_path": "~/Workbench/KindleClippings",
            "meta_path": "~/MetaLibrary",
            "notes_path": "~/Notes/Books",
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
        logger.info(f"Calibre Library {calibre.calibre_path}, loading metadata")
        calibre.load_calibre_library_metadata()
        logger.info("Calibre Library loaded")
        if do_export is True:
            logger.info(f"Calibre Library {calibre.calibre_path}, copying books")
            calibre.export_calibre_books(meta_path, dry_run=dry_run, delete=delete)
        if do_notes is True:
            logger.info(f"Calibre Library {calibre.calibre_path}, copying metadata")
            n, errs = calibre.export_calibre_metadata_to_markdown(
                notes_path, dry_run=dry_run, update_existing=False, delete=delete
            )
        else:
            n, errs = 0, 0
        logger.info(
            f"Calibre Library {calibre.calibre_path}, {n} book-descriptions exported, {errs} errors"
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
