import os
import logging
import xml.etree.ElementTree as ET
import datetime
from datetime import timezone
import time
import hashlib
import shutil
import json
import unicodedata
import zipfile
import zlib
import base64
from PIL import Image  # type: ignore
from bs4 import BeautifulSoup  # type:ignore ## pip install beautifulsoup4
from typing import TypedDict

from ebook_utils import sanitized_md_filename, progress_bar_string
from calibre_tools_localization import calibre_prefixes

# Disable MarkupResemblesLocatorWarning
from bs4 import MarkupResemblesLocatorWarning
import warnings

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)


class DocsEntry(TypedDict):
    path: str
    name: str
    size: int
    hash: str
    hash_algo: str
    mod_time: float
    ref_name: str


class CalibreLibEntry(TypedDict):
    title: str
    title_sort: str
    short_title: str
    short_folder: str
    description: str
    creators: list[str]
    series: str
    subjects: list[str]
    languages: list[str]
    publisher: str
    identifiers: list[str]
    uuid: str
    calibre_id: str
    publication_date: str
    date_added: str
    context: str
    cover: str
    doc_text: str
    docs: list[DocsEntry]
    formats: list[str]
    repo_path: list[str]


class RepoState(TypedDict):
    lib_entries: list[CalibreLibEntry]
    sequence_number: int
    timestamp: str

    
class CalibreTools:
    """ "Tools to manage Calibre library metadata and export to markdown

    - The library uses the metadata.opf files in the Calibre library to extract metadata
    - It can export books of given format as folder structure
    - It can export metadata to markdown files so that the library of books can be referred to in markdown notes collections
    """

    def __init__(self, calibre_path: str, calibre_library_name: str="Calibre_Library"):
        self.log: logging.Logger = logging.getLogger("CalibreTools")
        self.calibre_library_name: str = calibre_library_name
        cal_path = os.path.expanduser(calibre_path)
        self.sequence_number: int = 0
        self.lib_entries: list[CalibreLibEntry] = []
        self.old_lib_entries: list[CalibreLibEntry] = []
        if not os.path.exists(cal_path):
            self.log.error(f"Calibre path does not exist: {cal_path}")
            raise ValueError(f"Calibre path does not exist: {cal_path}")
        if not os.path.exists(os.path.join(cal_path, "metadata.db")):
            self.log.error(f"Error: Calibre metadata.db does not exist at {cal_path}")
            raise ValueError(f"Calibre metadata.db does not exist at {cal_path}")
        self.calibre_path: str = cal_path

    @staticmethod
    def _get_sha256(file_path: str) -> str:
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    @staticmethod
    def _get_crc32(file_path: str) -> int:
        crc32 = 0
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                crc32 = crc32 ^ zlib.crc32(chunk)
        return crc32

    @staticmethod
    def _is_number(s: str) -> bool:
        roman = True
        arabic = True
        for c in s.strip():
            if c not in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
                arabic = False
            if c not in ["I", "V", "X", "L", "C", "D", "M"]:
                roman = False
            if not arabic and not roman:
                return False
        return True

    @staticmethod
    def _clean_filename(s: str) -> str:
        bad_chars = ["<", ">", ":", '"', "/", "\\", "|", "?", "*"]
        for c in bad_chars:
            s = s.replace(c, ",")
        s = s.replace("__", "_")
        s = s.replace(" _ ", ", ")
        s = s.replace("_ ", ", ")
        s = s.replace("  ", " ")
        s = s.replace("  ", " ")
        s = s.replace(",,", ",")
        s = s.replace(" ,", " ")
        s = s.replace("  ", " ")
        s = s.replace("  ", " ")
        s = s.strip()
        s = unicodedata.normalize("NFC", s)
        return s

    def load_calibre_library_metadata(self, progress:bool=False, use_sha256:bool=False, load_text:bool=True):
        self.lib_entries = []

        total_entries = 0
        latest_mod_time = None
        if progress is True:
            for root, _dirs, files in os.walk(self.calibre_path):
                if ".caltrash" in root or ".calnotes" in root:
                    continue
                for file in files:
                    if file == "metadata.opf":
                        total_entries += 1
            if total_entries == 0:
                self.log.error("No metadata.opf files found in Calibre library")
                progress = False

        current_entry = 0
        start_time = time.time()
        mean_time_per_doc = 0
        for root, _dirs, files in os.walk(self.calibre_path):
            if ".caltrash" in root or ".calnotes" in root:
                continue
            for file in files:
                if file == "metadata.opf":

                    if progress is True:
                        current_entry += 1
                        progress_bar = progress_bar_string(current_entry, total_entries)
                        elapsed_time = time.time() - start_time
                        time_per_doc = elapsed_time / current_entry
                        if mean_time_per_doc == 0:
                            mean_time_per_doc = time_per_doc
                        else:
                            n0 = current_entry
                            if n0 > 500:
                                n0 = 500
                            f1 = 1.0 / n0
                            f2 = 1.0 - f1
                            mean_time_per_doc = f2 * mean_time_per_doc + f1 * time_per_doc
                        remaining_time = (total_entries - current_entry) * mean_time_per_doc
                        print(
                            f"{progress_bar} {current_entry}/{total_entries}, dt={mean_time_per_doc:.4f}, remaining: {remaining_time:.1f} sec.    ", end="\r"
                        )

                    title = ""
                    title_sort = ""
                    description = ""
                    creators = []
                    series = ""
                    subjects = []
                    languages = []
                    publisher = ""
                    identifiers: list[str] = []
                    uuid = ""
                    calibre_id = ""
                    pub_date = ""
                    date_added = ""

                    # cover = None
                    docs = []

                    # Read metadata.opf into python object using etree ET
                    filename = os.path.join(root, file)
                    # Get modification time
                    mod_time = os.path.getmtime(filename)
                    if latest_mod_time is None or mod_time > latest_mod_time:
                        latest_mod_time = mod_time
                    root_xml = ET.parse(filename).getroot()
                    # Namespace map
                    ns = {
                        "opf": "http://www.idpf.org/2007/opf",
                        "dc": "http://purl.org/dc/elements/1.1/",
                    }
                    # Extract metadata
                    metadata = root_xml.find("opf:metadata", ns)

                    if metadata is None:
                        self.log.error(f"No metadata found in OPF file for: {filename}")
                        continue

                    title_md = metadata.find("dc:title", ns)
                    title: str = str(title_md.text) if title_md is not None else ""
                    description_md = metadata.find("dc:description", ns)
                    description: str = str(description_md.text) if description_md is not None else ""

                    # creator = metadata.find("dc:creator", ns)
                    # creators = creator.text.split(", ") if creator is not None else []
                    # Get all authors from 'role': <dc:creator opf:file-as="Berlitz, Charles &amp; Moore, William L." opf:role="aut">Charles Berlitz</dc:creator>
                    # id.attrib["{http://www.idpf.org/2007/opf}scheme"]
                    creators: list[str] = []
                    for creator in metadata.findall("dc:creator", ns):
                        if "{http://www.idpf.org/2007/opf}role" in creator.attrib:
                            if (
                                creator.attrib["{http://www.idpf.org/2007/opf}role"]
                                == "aut"
                            ):
                                if isinstance(creator.text, str) and "," in creator.text:
                                    self.log.error(
                                        f"Author name contains comma: {creator.text}"
                                    )
                                creators.append(str(creator.text))

                    subjects_md = metadata.findall("dc:subject", ns)
                    subjects: list[str] = [str(subject.text) for subject in subjects_md]
                    languages_md = metadata.findall("dc:language", ns)
                    languages: list[str] = [str(language.text) for language in languages_md]
                    uuids_md = metadata.findall("dc:identifier", ns)
                    uuid: str = ""
                    calibre_id: str = ""
                    for u in uuids_md:
                        if "id" not in u.attrib:
                            continue
                        if u.attrib["id"] == "calibre_id":
                            calibre_id = str(u.text)
                        if u.attrib["id"] == "uuid_id":
                            uuid = str(u.text)

                    publisher_md = metadata.find("dc:publisher", ns)
                    publisher: str = str(publisher_md.text) if publisher_md is not None else ""
                    date_md = metadata.find("dc:date", ns)
                    date: str = str(date_md.text) if date_md is not None else ""
                    # convert to datetime, add utc timezone
                    if date != "":
                        pub_date = (
                            datetime.datetime.strptime(date, "%Y-%m-%dT%H:%M:%S%z")
                            .replace(tzinfo=timezone.utc)
                            .isoformat()
                        )

                    series: str = ""
                    date_added: str = ""
                    title_sort: str = ""
                    timestamp: str = ""
                    for meta in metadata.findall("opf:meta", ns):
                        if "name" in meta.attrib:
                            if meta.attrib["name"] == "calibre:series":
                                series = meta.attrib["content"]
                            if meta.attrib["name"] == "calibre:timestamp":
                                timestamp = str(meta.attrib["content"])
                                # timestamp can be 2023-11-11T17:03:48.214591+00:00 or 2023-11-11T17:03:48+00:00
                                timestamp = timestamp.split(".")[0]
                                if timestamp.endswith("+00:00"):
                                    date_added_dt = datetime.datetime.strptime(
                                        timestamp, "%Y-%m-%dT%H:%M:%S%z"
                                    )
                                else:
                                    date_added_dt = datetime.datetime.strptime(
                                        timestamp, "%Y-%m-%dT%H:%M:%S"
                                    )

                                date_added = date_added_dt.replace(
                                    tzinfo=timezone.utc
                                ).isoformat()
                            if meta.attrib["name"] == "calibre:title_sort":
                                title_sort = meta.attrib["content"]
                                for (
                                    lang
                                ) in (
                                    calibre_prefixes
                                ):  # remove localized prefixes ", The", ", Der", etc. (curr: DE, EN)
                                    prefixes = calibre_prefixes[lang]["prefixes"]
                                    for prefix in prefixes:
                                        ending = f", {prefix}"
                                        if title_sort.endswith(ending):
                                            title_sort = title_sort[: -len(ending)]
                                            break
                                # Check if starts with lowercase
                                if title_sort[0].islower():
                                    # check if second character is uppercase (iPad, jQuery, etc.)
                                    if len(title_sort) > 1 and title_sort[1].islower():
                                        self.log.warning(
                                            f"Shortened title starts with lowercase: {title_sort}, consider fixing!"
                                        )
                                        # title_sort = title_sort[0].upper() + title_sort[1:]  # automatic fixing can go wrong (jQuery, etc.)
                    identifiers = []
                    # Find records of type:
                    # <dc:identifier opf:scheme="MOBI-ASIN">B0BTX2378L</dc:identifier>
                    for id in metadata.findall("dc:identifier", ns):
                        # self.log.info(f"ID: {id.attrib} {id.text}")
                        if "{http://www.idpf.org/2007/opf}scheme" in id.attrib:
                            scheme = id.attrib["{http://www.idpf.org/2007/opf}scheme"]
                            sid = id.text
                            if scheme not in ["calibre", "uuid"]:
                                identifiers.append(f"{scheme}/{sid}")
                                # self.log.info(f"{title} Identifier: {scheme}: {sid}")
                    entry: CalibreLibEntry = {
                        "title": title,
                        "title_sort": title_sort,
                        "short_title": "",
                        "short_folder": "",
                        "description": description,
                        "creators": creators,
                        "series": series,
                        "subjects": subjects,
                        "languages": languages,
                        "publisher": publisher,
                        "identifiers": identifiers,
                        "uuid": uuid,
                        "calibre_id": calibre_id,
                        "publication_date": pub_date,
                        "date_added": date_added,
                        "context": f"Books/{series}",
                        "cover": "",
                        "doc_text": "",
                        "docs": [],
                        "formats": [],
                        "repo_path": [],
                    }
                    if "cover.jpg" in files:
                        entry["cover"] = os.path.join(root, "cover.jpg")
                    # check if pdf, epub, md, text files exist in this directory
                    exts: list[str] = [".pdf", ".epub", ".md", ".txt"]
                    docs: list[DocsEntry] = []
                    formats: list[str] = []
                    doc_text: str = ""
                    for doc in files:
                        if doc.endswith(tuple(exts)):
                            doc_full_name = os.path.join(root, doc)
                            # add extension to formats;
                            extension = doc.split(".")[-1]
                            formats.append(extension)
                            if load_text is True and extension == "txt":
                                with open(doc_full_name, "r") as f:
                                    doc_text = f.read()
                            size = os.path.getsize(doc_full_name)
                            mod_time: float = os.path.getmtime(doc_full_name)
                            if mod_time > latest_mod_time:
                                latest_mod_time = mod_time
                            if use_sha256 is False:
                                hash = str(CalibreTools._get_crc32(doc_full_name))
                                hash_algo = "crc32"
                            else:
                                hash = CalibreTools._get_sha256(doc_full_name)
                                hash_algo = "sha256"
                            docs.append(
                                {
                                    "path": doc_full_name,
                                    "name": doc,
                                    "size": size,
                                    "hash": hash,
                                    "hash_algo": hash_algo,
                                    "mod_time": mod_time,
                                    "ref_name": "",
                                }
                            )
                    entry["doc_text"] = doc_text
                    entry["docs"] = docs
                    entry["formats"] = formats
                    short_title = entry["title_sort"]

                    title = entry["title"]
                    # If title ends with roman or arabic numerals, store them as postfix:
                    endfix = ""
                    efs = title.split(" ")
                    for ef in efs:
                        if CalibreTools._is_number(ef):
                            endfix = ef
                            break
                        else:
                            ef = short_title.split(" ")[-1]
                            if CalibreTools._is_number(ef):
                                endfix = ef
                                break

                    short_title = CalibreTools._clean_filename(short_title)
                    max_title_len = 70
                    min_title_len = 30
                    if len(short_title) > max_title_len:
                        chars = [",", ".", "-", ":", ";"]
                        p = min(
                            (
                                short_title.find(c, min_title_len)
                                for c in chars
                                if short_title.find(c) != -1
                            ),
                            default=-1,
                        )
                        if p > min_title_len and p < max_title_len:
                            short_title = short_title[:p]
                        else:
                            p = short_title.find(" ", min_title_len, max_title_len)
                            if p == -1:
                                p = short_title.find("་", min_title_len, max_title_len)
                                if p != -1:
                                    p = p + 1  # add 1 to include the tsheg
                            if p == -1:
                                short_title = short_title[:max_title_len]
                            else:
                                short_title = short_title[:p]

                    short_title = short_title.strip()
                    if endfix != "":
                        if endfix not in short_title:
                            short_title = f"{short_title} {endfix}"
                    author = CalibreTools._clean_filename(entry["creators"][0])
                    short_title = f"{short_title.strip()} - {author}"

                    entry["short_title"] = short_title
                    entry["short_folder"] = f"{entry['series']}"
                    # Check if combination of short_title and short_folder is unique:
                    cmp = f"{entry['short_folder']}/{entry['short_title']}"
                    for ent in self.lib_entries:
                        test = f"{ent['short_folder']}/{ent['short_title']}"
                        if test == cmp:
                            self.log.error(
                                f"Duplicate found: there are two instances for: {cmp}, cannot continue due to ambiguity, please rename books"
                            )
                            self.log.error(f"  {test}")
                            self.log.error(f"  {cmp}")
                            exit(-1)
                    self.lib_entries.append(entry)
                    self.log.debug(f"Added entry: {entry['title']}")
                    # if max_entries is not None and len(self.lib_entries) >= max_entries:
                    #     self.log.warning(f"Reached max entries {max_entries}")
                    #     return latest_mod_time
        return latest_mod_time

    def check_epub_calibre_bookmarks(self, epub_path: str, entry, dry_run:bool=False):
        # Open epub and look if META-INF/calibre_bookmarks.txt exists
        # if yes, parse it (base64/json) and (TBD) add to entry
        with zipfile.ZipFile(epub_path, "r") as z:
            try:
                reader = z.open("META-INF/calibre_bookmarks.txt")
            except Exception as e:
                self.log.debug(f"No calibre_bookmarks for {entry['title']}: {e}")
                entry["calibre_bookmarks"] = []
                return
            bookmarks_base64 = reader.read().decode("utf-8")
            start_token = "encoding=json+base64:\n"
            if bookmarks_base64.startswith(start_token):
                bookmarks_base64 = (
                    bookmarks_base64.replace(start_token, "")
                    .replace("\n", "")
                    .replace("\r", "")
                )
                try:
                    bookmarks_jsonstr = base64.b64decode(bookmarks_base64)
                except Exception as e:
                    self.log.error(
                        f"Error decoding base64 calibre-bookmarks for {entry['title']}, {bookmarks_base64}: {e}"
                    )
                    entry["calibre_bookmarks"] = []
                    reader.close()
                    return
                try:
                    bookmarks = json.loads(bookmarks_jsonstr)
                except Exception as e:
                    self.log.error(
                        f"Error parsing json calibre-bookmarks for {entry['title']}, {bookmarks_jsonstr}: {e}"
                    )
                    entry["calibre_bookmarks"] = []
                    reader.close()
                    return
                if isinstance(bookmarks, list):
                    entry["calibre_bookmarks"] = bookmarks
                    if dry_run is False:
                        self.log.info(
                            f"TODO: sync calibre-bookmarks for {entry['title']}"
                        )
                    else:
                        self.log.info(
                            f"Would sync calibre-bookmarks for {entry['title']}"
                        )
                else:
                    self.log.error(
                        f"Calibre-Bookmarks of {entry['title']} not in expected format: {bookmarks_jsonstr}, expected list"
                    )
                    entry["calibre_bookmarks"] = []
                reader.close()
            else:
                self.log.error(
                    f"Calibre-Bookmarks of {entry['title']} not in expected format: {bookmarks_base64}"
                )
                entry["calibre_bookmarks"] = []
                reader.close()

    def export_calibre_books(
        self,
        target_path: str,
        format: list[str],
        dry_run: bool = False,
        delete: bool = False,
        vacuum: bool = False,
    ):
        target = os.path.expanduser(target_path)
        self.old_lib_entries, self.sequence_number = self.load_state(target)
        if len(self.lib_entries) == 0:
            self.log.error("No library entries found, cannot export")
            return 0, 0, 0
        updated = False
        new_docs = 0
        upd_docs = 0
        debris = 0
        upd_doc_names = []
        if not os.path.exists(target) and dry_run is not True:
            self.log.info(f"Creating target path {target}")
            os.makedirs(target)
        # Enumerate all files in target:
        target_existing: list[str] = []
        koreader_metadata = {}
        for root, dirs, files in os.walk(target):
            # if root is a dot dir, ignore
            if os.path.basename(root).startswith("."):
                if vacuum is True:
                    self.log.info(f"Ignoring dot folder {root}")
                continue
            if os.path.basename(root).endswith(".sdr"):
                if vacuum is True:
                    self.log.info(f"Skipping koreader metadata folder {root}")
                title = os.path.basename(root)[:-4]
                koreader_metadata[title] = {
                    "folder": root,
                    "formats": [],
                    "metadata": [],
                }
                epub_meta = os.path.join(root, "metadata.epub.lua")
                # XXX parse metadata for highlights, notes
                if os.path.exists(epub_meta):
                    koreader_metadata[title]["formats"].append("epub")
                    koreader_metadata[title]["metadata"].append({"epub": {}})
                pdf_meta = os.path.join(root, "metadata.pdf.lua")
                # XXX parse metadata for highlights, notes
                if os.path.exists(pdf_meta):
                    koreader_metadata[title]["formats"].append("pdf")
                    koreader_metadata[title]["metadata"].append({"pdf": {}})
                continue
            for file in files:
                if file in ["repo_state.json", "annotations.json"]:
                    continue
                filename = os.path.join(root, file)
                # ignore dot files
                if file.startswith("."):
                    continue
                # normalize filenames through unicodedata decomposition and composition to avoid iCloud+AFTP Unicode encoding issues
                filename = unicodedata.normalize("NFC", filename)
                target_existing.append(filename)
        for entry in self.lib_entries:
            folder = os.path.join(target, entry["short_folder"])
            folder_create = False
            if not os.path.exists(folder) and dry_run is not True:
                folder_create = True
            short_title = entry["short_title"]
            num_docs = len(entry["docs"])
            if num_docs == 0:
                self.log.error(f"No documents found for {short_title}")

            for index, doc in enumerate(entry["docs"]):
                ext = doc["name"].split(".")[-1]
                ref_name = f"{short_title}.{ext}"
                entry["docs"][index]["ref_name"] = ref_name
                if ext not in format:
                    continue
                if folder_create is True:
                    os.makedirs(folder)
                    folder_create = False
                doc_name = os.path.join(folder, ref_name)
                # normalize filenames through unicodedata decomposition and composition to avoid iCloud+AFTP Unicode encoding issues
                doc_name = unicodedata.normalize("NFC", doc_name)
                if not os.path.exists(doc_name):
                    # Copy file
                    updated = True
                    new_docs += 1
                    if ext == "epub":
                        self.check_epub_calibre_bookmarks(doc["path"], entry, dry_run)
                    if dry_run is False:
                        shutil.copy2(doc["path"], doc_name)
                        self.log.info(f"Copied {doc_name}")
                    else:
                        self.log.info(f"Would create {doc_name}")
                    if "repo_path" not in entry:
                        entry["repo_path"] = []
                    entry["repo_path"].append(doc_name)
                else:
                    doc_changed = False
                    if 'hash_algo' in doc and doc['hash_algo'] == 'crc32':
                        crc32 = CalibreTools._get_crc32(doc_name)
                        if str(crc32) != doc["hash"]:
                            self.log.warning(
                                f"CRC32 changed for {doc['name']} from {doc['hash']} to {crc32}"
                            )
                            doc_changed = True
                    else:
                        sha256 = CalibreTools._get_sha256(doc_name)
                        if sha256 != doc["hash"]:
                            self.log.warning(
                                f"SHA256 changed for {doc['name']} from {doc['hash']} to {sha256}"
                            )
                            doc_changed = True
                    if doc_changed:
                        updated = True
                        upd_docs += 1
                        upd_doc_names.append(doc_name)
                        if ext == "epub":
                            self.check_epub_calibre_bookmarks(
                                doc["path"], entry, dry_run
                            )
                        if dry_run is False:
                            shutil.copy2(doc["path"], doc_name)
                            self.log.warning(f"Updated **CHANGED** {doc_name}")
                        else:
                            self.log.warning(f"Would update **CHANGED** {doc_name}")
                    # Remove from target_existing
                    if doc_name in target_existing:
                        target_existing.remove(doc_name)
                    else:
                        self.log.error(
                            f"File {doc_name} not found in target_existing, Unicode encoding troubles in filename is most probable cause, manual cleanup required!"
                        )
                        self.log.warning(
                            "On macOS, filesystems that are unaware of upper/lowercase can cause this issue on renaming files with only case changes. Simply run twice!"
                        )
                        updated = True
        if len(target_existing) > 0:
            self.log.warning("Found files in target that are not in library:")
            updated = True
            for file in target_existing:
                if delete is True:
                    if dry_run is False:
                        os.remove(file)
                        self.log.warning(f"Deleted {file}")
                    else:
                        self.log.warning(f"Would delete {file}")
        # check if koreader metadata folders (.sdr) without corresponding library entry exist
        valid_books: list[str] = []
        sdr_debris: list[str] = []
        for entry in self.lib_entries:
            valid_books.append(entry["short_title"])
        for title in koreader_metadata:
            if title not in valid_books:
                sdr_debris.append(title)
        if len(sdr_debris) > 0:
            updated = True
            for title in sdr_debris:
                if delete is True:
                    if dry_run is False:
                        shutil.rmtree(koreader_metadata[title]["folder"])
                        self.log.warning(
                            f"Deleted obsolete metadata folder {koreader_metadata[title]['folder']}"
                        )
                    else:
                        self.log.warning(
                            f"Would delete obsolete metadata folder {koreader_metadata[title]['folder']}"
                        )
        else:
            self.log.debug(
                "No Koreader metadata folders without corresponding library entry found"
            )
        # Enumerate all folders, remove empty folders
        debris = len(target_existing)
        for root, dirs, files in os.walk(target, topdown=False):
            if len(files) == 0 and len(dirs) == 0:
                debris += 1
                if delete is True:
                    if dry_run is False:
                        # get folder name without path:
                        folder = os.path.basename(root)
                        if folder.startswith("."):  # don't kill .dot folders!
                            continue
                        os.rmdir(root)
                        self.log.warning(f"Removed empty folder {root}")
                    else:
                        self.log.warning(f"Would remove empty folder {root}")
        update_state = False
        if updated:
            self.log.info(
                f"Updated, new: {new_docs}, updated: {upd_docs}, debris: {debris}"
            )
            if len(upd_doc_names) > 0:
                self.log.info("Updated files:")
                for doc in upd_doc_names:
                    self.log.info(doc)
            if dry_run is False:
                update_state = True
        else:
            self.log.info("No updates and no new files found, nothing to do.")
        if update_state is True or self.sequence_number == 0:
            self.sequence_number = self.save_state(
                target, self.lib_entries, self.sequence_number
            )
            self.log.info(
                f"Saved state to {target}/repo_state.json, sequence number: {self.sequence_number}"
            )

        return new_docs, upd_docs, debris

    def save_state(self, target_path: str, lib_entries: list[CalibreLibEntry], sequence_number:int) -> int:
        repo_state_filename = os.path.join(target_path, "repo_state.json")
        sequence_number = sequence_number + 1
        repo_state: RepoState = {
            "lib_entries": lib_entries,
            "sequence_number": sequence_number,
            "timestamp": datetime.datetime.now(tz=timezone.utc).isoformat(),
        }
        with open(repo_state_filename, "w") as f:
            json.dump(repo_state, f, indent=4)
        return sequence_number

    def load_state(self, target_path: str) -> tuple[list[CalibreLibEntry], int]:
        repo_state_filename = os.path.join(target_path, "repo_state.json")
        if not os.path.exists(repo_state_filename):
            self.log.error(f"State file not found at {repo_state_filename}")
            return [], 0
        with open(repo_state_filename, "r") as f:
            repo_state: RepoState = json.load(f)
            lib_entries: list[CalibreLibEntry] = repo_state["lib_entries"]
            sequence_number: int = repo_state["sequence_number"]
            self.log.info(
                f"Loaded state from {repo_state_filename}, sequence number: {sequence_number}"
            )
        return lib_entries, sequence_number

    def check_state_change(self, target_path: str) -> bool:
        repo_state_filename = os.path.join(target_path, "repo_state.json")
        if not os.path.exists(repo_state_filename):
            self.log.error(f"State file not found at {repo_state_filename}")
            return False
        try:
            with open(repo_state_filename, "r") as f:
                repo_state: RepoState = json.load(f)
                sequence_number: int = repo_state["sequence_number"]
                if sequence_number != self.sequence_number:
                    self.log.warning(
                        f"repo_state.json sequence number changed from {sequence_number} to {self.sequence_number}"
                    )
                    return True
        except Exception as e:
            self.log.error(f"Error reading repo_state.json: {e}")
        return False

    def _gen_thumbnail(
        self, image_path: str, thumb_dir: str, thumb_dir_full: str, uuid: str, size: tuple[int, int]=(128, 128), force:bool=False
    ) -> str:
        dest_path_full = os.path.join(thumb_dir_full, uuid + ".jpg")
        dest_path_rel = os.path.join(thumb_dir, uuid + ".jpg")

        if os.path.exists(dest_path_full) and force is False:
            # self.log.info(f"Thumbnail {dest_path_full} already exists")
            return dest_path_rel

        with Image.open(image_path) as im:
            im.thumbnail(size)
            im.save(dest_path_full, "JPEG")
        return dest_path_rel

    def _gen_md_calibre_link(self, id:str ) -> str:
        # Example: calibre://show-book/_hex_-43616c696272655f4c696272617279/1515
        hex_name = "".join([hex(ord(c))[2:] for c in self.calibre_library_name])
        link = f"calibre://show-book/_hex_-{hex_name}/{id}"
        return link

    def export_calibre_metadata_to_markdown(
        self,
        notes,
        output_path:str,
        cover_rel_path:str | None = None,
        dry_run: bool = False,
        delete: bool = False,
    ) -> tuple[int, int, int]:
        output_path = os.path.expanduser(output_path)
        if not os.path.exists(output_path) and dry_run is not True:
            self.log.info(f"Creating output path {output_path}")
            os.makedirs(output_path)
        if cover_rel_path is None:
            cover_rel_path = "Covers"
        cover_full_path = os.path.join(output_path, cover_rel_path)
        if not os.path.exists(cover_full_path) and dry_run is not True:
            os.makedirs(cover_full_path)
        n = 0
        errs = 0
        content_updates = 0

        if notes.notes_books_folder != output_path:
            self.log.error(
                f"Notes books folder {notes.notes_books_folder} does not match output_path {output_path}"
            )
            return 0, 1, content_updates

        existing_notes_filenames = {}
        existing_notes_uuids = {}
        for note_filename in notes.notes:
            if note_filename.startswith(output_path):
                uuid = None
                if (
                    "metadata" in notes.notes[note_filename]
                    and "uuid" in notes.notes[note_filename]["metadata"]
                ):
                    uuid = notes.notes[note_filename]["metadata"]["uuid"]
                if uuid is not None:
                    existing_notes_filenames[note_filename] = uuid
                    existing_notes_uuids[uuid] = note_filename
                else:
                    self.log.error(
                        f"Note {note_filename} does not have a UUID, ignoring"
                    )

        for entry in self.lib_entries:
            mandatory_fields = [
                "title",
                "title_sort",
                "creators",
                "uuid",
                "publication_date",
                "date_added",
            ]
            for field in mandatory_fields:
                if field not in entry.keys():
                    errs += 1
                    print(f"Missing field {field} in entry {entry}")
                    continue
            sanitized_title = sanitized_md_filename(entry["title"])
            md_filename = os.path.join(output_path, f"{sanitized_title}.md")
            title = '"' + entry["title"].replace('"', "'") + '"'
            md = f"---\ncreation: {entry['date_added']}\ntitle: {title}\nuuid: {entry['uuid']}\nauthors:\n"
            # foot_tags=''
            foot_authors = ""
            authors = entry["creators"]
            for ind, author in enumerate(authors):
                md += f"  - {author}\n"
                if ind == len(authors) - 1:
                    foot_authors += f"[[{author}]]"
                else:
                    foot_authors += f"[[{author}]], "
            special_fields = [
                "cover",
                "comments",
                "tags",
                "formats",
                "description",
                "subjects",
                "series",
                "name",
                "docs",
                "short_title",
                "short_folder",
                "languages",
                "calibre_id",
                "identifiers",
                "publication_date",
            ]
            md += "tags:\n  - Library/Calibre\n"
            if "series" in entry.keys() and entry["series"] != "":
                ser = entry["series"].replace(" ", "_")
                md += f"  - Series/{ser}\n"
            if "subjects" in entry.keys() and len(entry["subjects"]) > 0:
                tags = entry["subjects"]
                for tag in tags:
                    filt_tag = tag.strip().replace(" ", "_")
                    md += f"  - {filt_tag}\n"
            if "formats" in entry.keys() and len(entry["formats"]) > 0:
                md += "formats:\n"
                formats = entry["formats"]
                for format in formats:
                    md += f"  - {format.strip()}\n"
            if (
                "publication_date" in entry.keys()
                and entry["publication_date"] != ""
            ):
                md += f"publication_date: {entry['publication_date']}\n"
            if "languages" in entry.keys() and len(entry["languages"]) > 0:
                md += "languages:\n"
                languages = entry["languages"]
                for lang in languages:
                    md += f"  - {lang.strip()}\n"
            if "identifiers" in entry.keys() and len(entry["identifiers"]) > 0:
                md += "identifiers:\n"
                ids = entry["identifiers"]
                for id in ids:
                    md += f"  - {id.strip()}\n"
            if "calibre_id" in entry.keys() and entry["calibre_id"] != "":
                md += f"calibre_id: {entry['calibre_id']}\n"
            string_fields = ["title_sort", "publisher"]
            for field in entry.keys():
                if (
                    field not in mandatory_fields
                    and field not in special_fields
                    and entry[field] is not None
                ):
                    if field in string_fields:
                        fld: str = str(entry[field]).replace('"', "'")
                        md += f'{field}: "{fld}"\n'
                    else:
                        md += f"{field}: {entry[field]}\n"
            md += "---\n"

            md += f"\n# {entry['title']}\n\n"
            md += "_by "
            first = True
            for author in entry["creators"]:
                if first:
                    first = False
                else:
                    md += ", "
                md += f"{author}"
            md += "_\n\n"
            if "calibre_id" in entry.keys() and entry["calibre_id"] != "":
                md += f"[Calibre-link]({self._gen_md_calibre_link(entry['calibre_id'])})\n\n"
            if "cover" in entry.keys() and entry["cover"] != "":
                if dry_run is False:
                    cover_path = self._gen_thumbnail(
                        entry["cover"],
                        cover_rel_path,
                        cover_full_path,
                        entry["uuid"],
                        force=False,
                    )
                    md += f"![{sanitized_title}]({cover_path})\n\n"
            if "description" in entry.keys() and entry["description"] != "":
                html_text = entry["description"]
                # Convert HTML to markdown
                md_tokens = [
                    ("<h1>", "&num; "),
                    ("<h2>", "&num;&num; "),
                    ("<h3>", "&num;&num;&num; "),
                    ("<h4>", "&num;&num;&num;&num; "),
                    ("</h1>", "\n\n"),
                    ("</h2>", "\n\n"),
                    ("</h3>", "\n\n"),
                    ("</h4>", "\n\n"),
                    ("<em>", " *"),
                    ("</em>", "* "),
                    ("<strong>", "**"),
                    ("</strong>", "** "),
                    ("<p>", ""),
                    ("</p>", "\n\n"),
                    ("<br>", "\n\n"),
                    ("<br/>", "\n\n"),
                    ("<br />", "\n\n"),
                    ("<li>", "- "),
                    ("</li>", "\n"),
                    ("  ", " "),
                    ("  ", " "),
                ]
                for token in md_tokens:
                    html_text = html_text.replace(token[0], token[1])
                text = BeautifulSoup(
                    html_text, features="lxml"
                ).get_text()  # "html.parser").get_text()
                cmt = text.replace("\n", "\n> ")
                md += f"> {cmt}\n"
            # if len(foot_tags) > 3:
            #     md += f"\nTags: {foot_tags[:-2]}\n"
            if len(foot_authors) > 3:
                md += f"\nAuthors: {foot_authors}\n"

            uuid_exists = False
            if entry["uuid"] in notes.uuid_to_note_filename:
                uuid_exists = True
                if entry["uuid"] in existing_notes_uuids:
                    old_filename: str = notes.uuid_to_note_filename[entry["uuid"]]
                    if old_filename != md_filename:
                        self.log.warning(
                            f"Note {old_filename} was renamed to {md_filename}"
                        )
                        content_updates += notes.rename_note(
                            old_filename,
                            md_filename,
                            update_links=True,
                            dry_run=dry_run,
                        )
                        if old_filename in existing_notes_filenames:
                            del existing_notes_filenames[old_filename]
                        if entry["uuid"] in existing_notes_uuids:
                            del existing_notes_uuids[entry["uuid"]]
                    else:
                        # remove from existing_notes
                        if md_filename in existing_notes_filenames:
                            del existing_notes_filenames[md_filename]
                        if entry["uuid"] in existing_notes_uuids:
                            old_filename: str = existing_notes_uuids[entry["uuid"]]
                            del existing_notes_uuids[entry["uuid"]]
                            if old_filename != md_filename:
                                self.log.error(
                                    f"Note {old_filename} was renamed to {md_filename}, but not updated in existing_notes"
                                )
                                del existing_notes_filenames[old_filename]
            if os.path.exists(md_filename) is False and uuid_exists is False:
                if dry_run is False:
                    with open(md_filename, "w") as f:
                        f.write(md)
                else:
                    self.log.info(f"Would write file {md_filename}")
                n += 1
        if len(existing_notes_filenames) > 0:
            self.log.warning(
                f"Found {len(existing_notes_filenames)} existing notes that are not in the library"
            )
            for note in existing_notes_filenames:
                if delete is True:
                    if dry_run is False:
                        os.remove(note)
                        self.log.warning(f"Deleted {note}")
                    else:
                        self.log.warning(f"Would delete {note}")
        return n, errs, content_updates
