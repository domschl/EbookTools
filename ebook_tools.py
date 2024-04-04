import logging
from calibre_tools import CalibreTools

if __name__ == "__main__":
    # Init logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    calibre = CalibreTools(calibre_path="~/ReferenceLibrary/Calibre Library")
    logger.info(f"Calibre Library {calibre.calibre_path}, loading metadata")
    calibre.load_calibre_library_metadata()
    logger.info(f"Calibre Library {calibre.calibre_path}, copying books")
    # calibre.export_calibre_books("~/Downloads/Test/CalibreBooks")
    # logger.info(f"Calibre Library {calibre.calibre_path}, copying metadata")
    calibre.export_calibre_metadata_to_markdown("~/Downloads/Test/CalibreMetaData")
    logger.info(f"Calibre Library {calibre.calibre_path}, done")
