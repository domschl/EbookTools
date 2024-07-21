import os
import logging
import xml.etree.ElementTree as ET
import datetime
from datetime import timezone
import hashlib
import shutil
import json
import unicodedata

from PIL import Image
from bs4 import BeautifulSoup  ## pip install beautifulsoup4

from ebook_utils import sanitized_md_filename
from calibre_tools_localization import calibre_prefixes


class CalibreTools:
    """ "Tools to manage Calibre library metadata and export to markdown

    - The library uses the metadata.opf files in the Calibre library to extract metadata
    - It can export books of given format as folder structure
    - It can export metadata to markdown files so that the library of books can be referred to in markdown notes collections
    """

    def __init__(self, calibre_path, calibre_library_name="Calibre_Library"):
        self.log = logging.getLogger("CalibreTools")
        self.calibre_library_name = calibre_library_name
        cal_path = os.path.expanduser(calibre_path)
        self.sequence_number = 0
        self.lib_entries = []
        if not os.path.exists(cal_path):
            self.log.error(f"Calibre path does not exist: {cal_path}")
            raise ValueError(f"Calibre path does not exist: {cal_path}")
        if not os.path.exists(os.path.join(cal_path, "metadata.db")):
            self.log.error(f"Error: Calibre metadata.db does not exist at {cal_path}")
            raise ValueError(f"Calibre metadata.db does not exist at {cal_path}")
        self.calibre_path = cal_path

    @staticmethod
    def _get_sha256(file_path):
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    @staticmethod
    def _is_number(s):
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
    def _clean_filename(s):
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
        return s

    def load_calibre_library_metadata(self, max_entries=None):
        self.lib_entries = []
        for root, dirs, files in os.walk(self.calibre_path):
            if ".caltrash" in root or ".calnotes" in root:
                continue
            for file in files:
                if file == "metadata.opf":

                    title = None
                    title_sort = None
                    description = None
                    creators = []
                    series = None
                    subjects = []
                    languages = []
                    publisher = None
                    identifiers = []
                    uuid = None
                    calibre_id = None
                    pub_date = None
                    date_added = None
                    
                    # cover = None
                    docs = []

                    # Read metadata.opf into python object using etree ET
                    filename = os.path.join(root, file)
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

                    title = metadata.find("dc:title", ns)
                    title = title.text if title is not None else None
                    description = metadata.find("dc:description", ns)
                    description = description.text if description is not None else None

                    # creator = metadata.find("dc:creator", ns)
                    # creators = creator.text.split(", ") if creator is not None else []
                    # Get all authors from 'role': <dc:creator opf:file-as="Berlitz, Charles &amp; Moore, William L." opf:role="aut">Charles Berlitz</dc:creator>
# id.attrib["{http://www.idpf.org/2007/opf}scheme"]
                    creators = []
                    for creator in metadata.findall("dc:creator", ns):
                        if "{http://www.idpf.org/2007/opf}role" in creator.attrib:
                            if creator.attrib["{http://www.idpf.org/2007/opf}role"] == "aut":
                                if ',' in creator.text:
                                    self.log.error(f"Author name contains comma: {creator.text}")
                                creators.append(creator.text)
                    
                    subjects = metadata.findall("dc:subject", ns)
                    subjects = [subject.text for subject in subjects]
                    languages = metadata.findall("dc:language", ns)
                    languages = [language.text for language in languages]
                    uuids = metadata.findall("dc:identifier", ns)
                    uuid = None
                    calibre_id = None
                    for u in uuids:
                        if "id" not in u.attrib:
                            continue
                        if u.attrib["id"] == "calibre_id":
                            calibre_id = u.text
                        if u.attrib["id"] == "uuid_id":
                            uuid = u.text

                    publisher = metadata.find("dc:publisher", ns)
                    publisher = publisher.text if publisher is not None else None
                    date = metadata.find("dc:date", ns)
                    date = date.text if date is not None else None
                    # convert to datetime, add utc timezone
                    pub_date = (
                        datetime.datetime.strptime(date, "%Y-%m-%dT%H:%M:%S%z")
                        .replace(tzinfo=timezone.utc)
                        .isoformat()
                    )

                    series = None
                    date_added = None
                    title_sort = None
                    timestamp = None
                    for meta in metadata.findall("opf:meta", ns):
                        if "name" in meta.attrib:
                            if meta.attrib["name"] == "calibre:series":
                                series = meta.attrib["content"]
                            if meta.attrib["name"] == "calibre:timestamp":
                                timestamp = meta.attrib["content"]
                                # timestamp can be 2023-11-11T17:03:48.214591+00:00 or 2023-11-11T17:03:48+00:00
                                timestamp = timestamp.split(".")[0]
                                if timestamp.endswith("+00:00"):
                                    date_added = datetime.datetime.strptime(
                                        timestamp, "%Y-%m-%dT%H:%M:%S%z"
                                    )
                                else:
                                    date_added = datetime.datetime.strptime(
                                        timestamp, "%Y-%m-%dT%H:%M:%S"
                                    )

                                date_added = date_added.replace(
                                    tzinfo=timezone.utc
                                ).isoformat()
                            if meta.attrib["name"] == "calibre:title_sort":
                                title_sort = meta.attrib["content"]
                                for lang in calibre_prefixes:  # remove localized prefixes ", The", ", Der", etc. (curr: DE, EN)
                                    for prefix in calibre_prefixes[lang]["prefixes"]:
                                        ending = f", {prefix}"
                                        if title_sort.endswith(ending):
                                            title_sort = title_sort[:-len(ending)]
                                            break
                                # Check if starts with lowercase
                                if title_sort[0].islower():
                                    # check if second character is uppercase (iPad, jQuery, etc.)
                                    if len(title_sort)>1 and title_sort[1].islower():
                                        self.log.warning(f"Shortened title starts with lowercase: {title_sort}, consider fixing!")
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
                    entry = {
                        "title": title,
                        "title_sort": title_sort,
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
                    }
                    if "cover.jpg" in files:
                        entry["cover"] = os.path.join(root, "cover.jpg")
                    # check if pdf, epub, md, text files exist in this directory
                    exts = [".pdf", ".epub", ".md", ".txt"]
                    docs = []
                    formats = []
                    for doc in files:
                        if doc.endswith(tuple(exts)):
                            doc_full_name = os.path.join(root, doc)
                            # add extension to formats;
                            formats.append(doc.split(".")[-1])
                            # calculate size and hash (sha256):
                            size = os.path.getsize(doc_full_name)
                            hash = CalibreTools._get_sha256(doc_full_name)
                            docs.append(
                                {
                                    "path": doc_full_name,
                                    "name": doc,
                                    "size": size,
                                    "hash": hash,
                                }
                            )
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
                                    p = p + 1  #  add 1 to include the tsheg
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
                    duplicate = False
                    for ent in self.lib_entries:
                        test = f"{ent['short_folder']}/{ent['short_title']}"
                        if test == cmp:
                            duplicate = True
                            self.log.error(
                                f"Duplicate found: there are two instances for: {cmp}, cannot continue due to ambiguity, please rename books"
                            )
                            self.log.error(f"  {test}")
                            self.log.error(f"  {cmp}")
                            exit(-1)
                    self.lib_entries.append(entry)
                    self.log.debug(f"Added entry: {entry['title']}")
                    if max_entries is not None and len(self.lib_entries) >= max_entries:
                        self.log.warning(f"Reached max entries {max_entries}")
                        return self.lib_entries
        return self.lib_entries

    def export_calibre_books(
        self,
        target_path,
        format=["pdf", "epub", "md", "txt"],
        dry_run=False,
        delete=False,
    ):
        target = os.path.expanduser(target_path)
        self.old_lib_entries, self.sequence_number = self.load_state(target)
        if self.lib_entries is None or len(self.lib_entries) == 0:
            self.log.error("No library entries found, cannot export")
            return 0, 0, 0
        updated = False
        new_docs = 0
        upd_docs = 0
        debris = 0
        upd_doc_names = []
        if not os.path.exists(target) and not dry_run is True:
            self.log.info(f"Creating target path {target}")
            os.makedirs(target)
        # Enumerate all files in target:
        target_existing = []
        for root, dirs, files in os.walk(target):
            for file in files:
                # if root is a dot dir, ignore
                if os.path.basename(root).startswith("."):
                    continue
                if file == "repo_state.json":
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
            if not os.path.exists(folder) and not dry_run is True:
                os.makedirs(folder)
            short_title = entry["short_title"]
            num_docs = len(entry["docs"])
            if num_docs == 0:
                self.log.error(f"No documents found for {short_title}")

            for doc in entry["docs"]:
                ext = doc["name"].split(".")[-1]
                if ext not in format:
                    continue
                doc_name = os.path.join(folder, f"{short_title}.{ext}")
                # normalize filenames through unicodedata decomposition and composition to avoid iCloud+AFTP Unicode encoding issues
                doc_name = unicodedata.normalize("NFC", doc_name)
                if not os.path.exists(doc_name):
                    # Copy file
                    updated = True
                    new_docs += 1
                    if dry_run is False:
                        shutil.copy2(doc["path"], doc_name)
                        self.log.info(f"Copied {doc_name}")
                    else:
                        self.log.info(f"Would create {doc_name}")
                    if "repo_path" not in entry:
                        entry["repo_path"] = []
                    entry["repo_path"].append(doc_name)
                else:
                    # Check sha256:
                    sha256 = CalibreTools._get_sha256(doc_name)
                    if sha256 != doc["hash"]:
                        self.log.warning(
                            f"SHA256 changed for {doc['name']} from {doc['hash']} to {sha256}"
                        )
                        updated = True
                        upd_docs += 1
                        upd_doc_names.append(doc_name)
                        if dry_run is False:
                            shutil.copy2(doc["path"], doc_name)
                            self.log.warning(
                                f"Updated **CHANGED** {doc_name}"
                            )
                        else:
                            self.log.warning(
                                f"Would update **CHANGED** {doc_name}"
                            )
                    # Remove from target_existing
                    if doc_name in target_existing:
                        target_existing.remove(doc_name)
                    else:
                        self.log.error(
                            f"File {doc_name} not found in target_existing, Unicode encoding troubles in filename is most probable cause, manual cleanup required!"
                        )
                        self.log.warning("On macOS, filesystems that are unaware of upper/lowercase can cause this issue on renaming files with only case changes. Simply run twice!")
                        updated = True
        if len(target_existing) > 0:
            self.log.warning("Found files in target that are not in library:")
            updated = True
            for file in target_existing:
                if delete is True and dry_run is False:
                    os.remove(file)
                    self.log.warning(f"Deleted {file}")
                    # Check, if folder is empty, remove it
                    folder = os.path.dirname(file)
                else:
                    self.log.warning(f"Would delete {file}")
        # Enumerate all folders, remove empty folders
        debris = len(target_existing)
        for root, dirs, files in os.walk(target, topdown=False):
            if len(files) == 0 and len(dirs) == 0:
                debris += 1
                if delete is True and dry_run is False:
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
            self.sequence_number = self.save_state(target, self.lib_entries, self.sequence_number)
            self.log.info(f"Saved state to {target}/repo_state.json, sequence number: {self.sequence_number}    ")

        return new_docs, upd_docs, debris

    def save_state(self, target_path, lib_entries, sequence_number):
        repo_state_filename = os.path.join(target_path, "repo_state.json")
        sequence_number = sequence_number + 1
        repo_state = {
            "lib_entries": lib_entries,
            "sequence_number": sequence_number,
            "timestamp": datetime.datetime.now(tz=timezone.utc).isoformat(),
        }
        with open(repo_state_filename, "w") as f:
            json.dump(repo_state, f, indent=4)
        return sequence_number

    def load_state(self, target_path):
        repo_state_filename = os.path.join(target_path, "repo_state.json")
        if not os.path.exists(repo_state_filename):
            self.log.error(f"State file not found at {repo_state_filename}")
            return [], 0
        with open(repo_state_filename, "r") as f:
            repo_state = json.load(f)
            lib_entries = repo_state["lib_entries"]
            sequence_number = repo_state["sequence_number"]
            self.log.info(
                f"Loaded state from {repo_state_filename}, sequence number: {self.sequence_number}"
            )
        return lib_entries, sequence_number

    def check_state_change(self, target_path):
        repo_state_filename = os.path.join(target_path, "repo_state.json")
        if not os.path.exists(repo_state_filename):
            self.log.error(f"State file not found at {repo_state_filename}")
            return False
        try:
            with open(repo_state_filename, "r") as f:
                repo_state = json.load(f)
                sequence_number = repo_state["sequence_number"]
                if sequence_number != self.sequence_number:
                    self.log.warning(
                        f"repo_state.json sequence number changed from {sequence_number} to {self.sequence_number}"
                    )
                    return True
        except Exception as e:
            self.log.error(f"Error reading repo_state.json: {e}")
        return False
    
    def _gen_thumbnail(
        self, image_path, thumb_dir, thumb_dir_full, uuid, size=(128, 128), force=False
    ):
        dest_path_full = os.path.join(thumb_dir_full, uuid + ".jpg")
        dest_path_rel = os.path.join(thumb_dir, uuid + ".jpg")

        if os.path.exists(dest_path_full) and force is False:
            # self.log.info(f"Thumbnail {dest_path_full} already exists")
            return dest_path_rel

        with Image.open(image_path) as im:
            im.thumbnail(size)
            im.save(dest_path_full, "JPEG")
        return dest_path_rel

    def _gen_md_calibre_link(self, id) -> str:
        # Example: calibre://show-book/_hex_-43616c696272655f4c696272617279/1515
        hex_name = "".join([hex(ord(c))[2:] for c in self.calibre_library_name])
        link = f"calibre://show-book/_hex_-{hex_name}/{id}"
        return link

    def export_calibre_metadata_to_markdown(
        self, output_path, max_entries=None, cover_rel_path=None, update_existing=False, dry_run=False, delete=False
    ):
        output_path = os.path.expanduser(output_path)
        if not os.path.exists(output_path) and not dry_run is True:
            self.log.info(f"Creating output path {output_path}")
            os.makedirs(output_path)
        if cover_rel_path is None:
            cover_rel_path = "Covers"
        cover_full_path = os.path.join(output_path, cover_rel_path)
        if not os.path.exists(cover_full_path) and not dry_run is True:
            os.makedirs(cover_full_path)
        n = 0
        errs = 0
        existing_notes = []
        for root, dirs, files in os.walk(output_path):
            for file in files:
                if file.endswith(".md"):
                    existing_notes.append(os.path.join(root, file))

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
            if "series" in entry.keys() and entry["series"] is not None:
                ser = entry["series"].replace(" ", "_")
                md += f"  - Series/{ser}\n"
            if "subjects" in entry.keys() and entry["subjects"] is not None:
                tags = entry["subjects"]
                for tag in tags:
                    filt_tag = tag.strip().replace(" ", "_")
                    md += f"  - {filt_tag}\n"
            if "formats" in entry.keys() and entry["formats"] is not None:
                md += "formats:\n"
                formats = entry["formats"]
                for format in formats:
                    md += f"  - {format.strip()}\n"
            if (
                "publication_date" in entry.keys()
                and entry["publication_date"] is not None
            ):
                md += f"publication_date: {entry['publication_date']}\n"
            if "languages" in entry.keys() and entry["languages"] is not None:
                md += "languages:\n"
                languages = entry["languages"]
                for lang in languages:
                    md += f"  - {lang.strip()}\n"
            if "identifiers" in entry.keys() and entry["identifiers"] is not None:
                md += "identifiers:\n"
                ids = entry["identifiers"]
                for id in ids:
                    md += f"  - {id.strip()}\n"
            if "calibre_id" in entry.keys() and entry["calibre_id"] is not None:
                md += f"calibre_id: {entry['calibre_id']}\n"
            string_fields = ['title_sort', 'publisher']
            for field in entry.keys():
                if (
                    field not in mandatory_fields
                    and field not in special_fields
                    and entry[field] is not None
                ):
                    if field in string_fields:
                        md += f"{field}: \"{entry[field].replace('"', "'")}\"\n"
                    else:
                        md += f"{field}: {entry[field]}\n"
            md += "---\n"

            md += f"\n# {entry['title']}\n\n"
            md += f"_by "
            first = True
            for author in entry["creators"]:
                if first:
                    first = False
                else:
                    md += f", "
                md += f"{author}"
            md += "_\n\n"
            if "calibre_id" in entry.keys() and entry["calibre_id"] is not None:
                md += f"[Calibre-link]({self._gen_md_calibre_link(entry['calibre_id'])})\n\n"
            if "cover" in entry.keys() and entry["cover"] is not None:
                if dry_run is False:
                    cover_path = self._gen_thumbnail(
                        entry["cover"],
                        cover_rel_path,
                        cover_full_path,
                        entry["uuid"],
                        force=False,
                    ) 
                    md += f"![{sanitized_title}]({cover_path})\n\n"
            if "description" in entry.keys() and entry["description"] is not None:
                html_text = entry["description"]
                # Convert HTML to markdown
                md_tokens = [("<h1>", "&num; "), ("<h2>", "&num;&num; "), ("<h3>", "&num;&num;&num; "), ("<h4>", "&num;&num;&num;&num; "),
                             ("</h1>", "\n\n"), ("</h2>", "\n\n"), ("</h3>", "\n\n"), ("</h4>", "\n\n"), 
                             ("<em>", " *"), ("</em>", "* "), 
                             ("<strong>", "**"), ("</strong>", "** "),
                             ("<p>", ""), ("</p>", "\n\n"), ("<br>", "\n\n"), ("<br/>", "\n\n"), ("<br />", "\n\n"), 
                             ("<li>", "- "), ("</li>", "\n"),
                             ("  ", " "), ("  ", " ")]
                for token in md_tokens:
                    html_text = html_text.replace(token[0], token[1])
                text = BeautifulSoup(html_text, "html.parser").get_text()
                cmt = text.replace("\n", "\n> ")
                md += f"> {cmt}\n"
            # if len(foot_tags) > 3:
            #     md += f"\nTags: {foot_tags[:-2]}\n"
            if len(foot_authors) > 3:
                md += f"\nAuthors: {foot_authors}\n"
            if md_filename in existing_notes:
                # remove from existing_notes
                existing_notes.remove(md_filename)
            if os.path.exists(md_filename):
                if update_existing is True:
                    with open(md_filename, "r") as f:
                        existing_md = f.read()
                    if existing_md == md:
                        self.log.info(f"File {md_filename} already exists and is unchanged")
                    else:
                        diffs = self.notes_differ(existing_md, md)
                        if diffs > 0:
                            self.log.warning(
                                f"File {md_filename} already exists but is changed, {diffs} differences found, UPDATE NOT IMPLEMENTED"
                            )
                            errs += 1
                        else:
                            self.log.info("File {md_filename} has been updated, no significant change")
                    continue
                else:
                    continue
            else:
                if dry_run is False:
                    with open(md_filename, "w") as f:
                        f.write(md)
                else:
                    self.log.info(f"Would write file {md_filename}")
                n += 1
                if n == max_entries:
                    break
        if len(existing_notes)>0:
            self.log.warning(f"Found {len(existing_notes)} existing notes that are not in the library")
            for note in existing_notes:
                if delete is True:
                    if dry_run is False:
                        os.remove(note)
                        self.log.warning(f"Deleted {note}")
                    else:
                        self.log.warning(f"Would delete {note}")
        return n, errs
