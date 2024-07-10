import logging
import sys

from calibre_tools import CalibreTools
from kindle_tools import KindleTools

if __name__ == "__main__":
    # Init logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    dry_run = False
    delete = False
    do_export = False
    do_notes = False
    do_kindle = False

    args = sys.argv[1:]
    if "-d" in args:
        dry_run = True
        args.remove("-d")
    if "-x" in args:
        delete = True
        args.remove("-x")
    if "-h" in args:
        print("Usage: python ebook_tools.py [export] [notes] [kindle] [-d] [-x]")
        print("  export: export books from Calibre Library to MetaLibrary folder")
        print("  notes:  export metadata to Notes as Markdown files")
        print("  kindle: export Kindle clippings to Notes as Markdown files")
        print("  -d: dry run, do not copy files")
        print("  -x: delete files that are debris")
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

    if do_export is True or do_notes is True:
        calibre = CalibreTools(calibre_path="~/ReferenceLibrary/Calibre Library")
        logger.info(f"Calibre Library {calibre.calibre_path}, loading metadata")
        calibre.load_calibre_library_metadata()
        logger.info("Calibre Library loaded")
        if do_export is True:
            logger.info(f"Calibre Library {calibre.calibre_path}, copying books")
            calibre.export_calibre_books(
                "~/Downloads/MetaLibrary", dry_run=dry_run, delete=delete
            )
        if do_notes is True:
            logger.info(f"Calibre Library {calibre.calibre_path}, copying metadata")
            n, errs = calibre.export_calibre_metadata_to_markdown("~/Downloads/Books")
        else:
            n, errs = 0, 0
        logger.info(
            f"Calibre Library {calibre.calibre_path}, {n} book-descriptions exported, {errs} errors"
        )
    if do_kindle is True:
        kindle = KindleTools()
        if kindle.check_for_connected_kindle() is False:
            logger.error("No Kindle connected!")
            # clippings_text = kindle.get_clippings_text("~/Workbench/My Clippings_oasis.txt")
            clippings_text = kindle.get_clippings_text("~/Workbench/My Clippings.txt")
            # exit(1)
        else:
            clippings_text = kindle.get_clippings_text()
        clippings = kindle.parse_clippings(clippings_text)
        if clippings is None or len(clippings) == 0:
            logger.error("No clippings found, something went wrong...")
            exit(1)

        logger.info(f"Found {len(clippings)} clippings")
        for i in range(0, 10):
            print(clippings[i])
