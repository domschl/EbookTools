import logging
import pypdf
import zipfile
from lxml import etree

class Metadata:
    def __init__(self, filepath):
        self.filepath = filepath
        self.log = logging.getLogger(__name__)
        self.metadata = {}
        self.metadata_valid = False
        self._get()

    def _get(self):
        if self.filepath.endswith(".pdf"):
            self._get_pdf()
        elif self.filepath.endswith(".epub"):
            self._get_epub()
        else:
            self.log.error(f"Unsupported file type: {self.filepath}")
           
    def _get_pdf(self):
        reader = pypdf.PdfReader(self.filepath, strict=True)
        if reader.metadata is not None:
            for key in reader.metadata.keys():
                self.metadata[key] = reader.metadata[key]
            self.metadata_valid = True

    def _get_epub(self):
        def xpath(element, path):
            el = element.xpath(
                path,
                namespaces={
                    "n": "urn:oasis:names:tc:opendocument:xmlns:container",
                    "pkg": "http://www.idpf.org/2007/opf",
                    "dc": "http://purl.org/dc/elements/1.1/",
                },
            )
            if el and len(el) > 0:
                return el[0]
            return None

        # prepare to read from the .epub file
        try:
            zip_content = zipfile.ZipFile(self.filepath, "r")
        except zipfile.BadZipFile:
            self.log.error(f"Bad Zip file: {self.filepath}")
            self.metadata_valid = False
            exit(-1)
            return
      
        # find the contents metafile
        try:
            cfname = xpath(
                etree.fromstring(zip_content.read("META-INF/container.xml")),
                "n:rootfiles/n:rootfile/@full-path",
            )
        except Exception as e:
            self.log.error(f"Error reading container.xml or {self.filepath}: {e}")
            self.metadata_valid = False
            exit(-1)
            return
    
        # grab the metadata block from the contents metafile
        try:
            metadata = xpath(
                etree.fromstring(zip_content.read(cfname)), "/pkg:package/pkg:metadata"
            )
        except Exception as e:
            self.log.error(f"Error reading {cfname} in {self.filepath}: {e}")
            self.metadata_valid = False
            exit(-1)
            return

        # repackage the data
        md = {
            s: xpath(metadata, f"dc:{s}/text()")
            for s in ("title", "language", "creator", "date", "identifier")
        }
        # print(md)
        for key in md.keys():
            key = str(key)
            self.metadata[key] = md[key]
        self.metadata_valid = True
                    
    def get_metadata(self):
        if self.metadata_valid is False:
            self.log.error(f"Metadata not valid for {self.filepath}")
            return None
        else:
            return self.metadata
                    
    def set_metadata(self, metadata):
        for key in metadata:
            self.metadata[key] = metadata[key]
        self.metadata_valid = True    

    def write_metadata():
        if self.metadata_valid is False:
            self.log.error("Metadata not valid")
            return False
        if self.filepath.endswith(".pdf"):
            return self._write_pdf()
        else:
            self.log.error(f"Unsupported file type: {self.filepath}")
            return False

    def _write_pdf(self):
        if self.metadata_valid is False:
            self.log.error(f"Metadata not valid for {self.filepath}")
            return False
        reader = pypdf.PdfReader(self.filepath)
        writer = pdf.PdfWriter()
        for page in reader.pages:
            writer.add_page(page)
        writer.add_metadata(self.metadata)
        with open(self.filepath, "wb") as f:
            writer.write(f)
        return True
