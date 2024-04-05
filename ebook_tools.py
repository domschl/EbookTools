import logging
from calibre_tools import CalibreTools
from kindle_tools import KindleTools

if __name__ == "__main__":
    # Init logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    if False:
        calibre = CalibreTools(calibre_path="~/ReferenceLibrary/Calibre Library")
        logger.info(f"Calibre Library {calibre.calibre_path}, loading metadata")
        calibre.load_calibre_library_metadata()
        # logger.info(f"Calibre Library {calibre.calibre_path}, copying books")
        # calibre.export_calibre_books("~/Downloads/Test/CalibreBooks")
        logger.info(f"Calibre Library {calibre.calibre_path}, copying metadata")
        calibre.export_calibre_metadata_to_markdown("~/Downloads/Test/CalibreMetaData")
        logger.info(f"Calibre Library {calibre.calibre_path}, done")
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
