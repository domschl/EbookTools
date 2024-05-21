import os
import logging
import xml.etree.ElementTree as ET
import datetime
from datetime import timezone
import hashlib
import shutil
import json
from PIL import Image
import unicodedata

from ebook_utils import sanitized_md_filename


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

                    creator = metadata.find("dc:creator", ns)
                    creators = creator.text.split(", ") if creator is not None else []
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
        updated = False
        new_docs = 0
        upd_docs = 0
        upd_doc_names = []
        if not os.path.exists(target):
            self.log.info(f"Creating target path {target}")
            os.makedirs(target)
        # Enumerate all files in target:
        target_existing = []
        for root, dirs, files in os.walk(target):
            for file in files:
                filename = os.path.join(root, file)
                # normalize filenames through unicodedata decomposition and composition to avoid iCloud+AFTP Unicode encoding issues
                filename = unicodedata.normalize("NFC", filename)
                target_existing.append(filename)
        for entry in self.lib_entries:
            folder = os.path.join(target, entry["short_folder"])
            if not os.path.exists(folder):
                os.makedirs(folder)
            short_title = entry["short_title"]
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
                        self.log.info(f"Copied {doc['name']} to {folder}")
                    else:
                        self.log.info(f"Would copy {doc['name']} to {folder}")
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
                                f"Updated **CHANGED** {doc['name']} in {folder}"
                            )
                        else:
                            self.log.warning(
                                f"Would update **CHANGED** {doc['name']} in {folder}"
                            )
                    # Remove from target_existing
                    if doc_name in target_existing:
                        target_existing.remove(doc_name)
                    else:
                        self.log.error(
                            f"File {doc_name} not found in target_existing, Unicode encoding troubles in filename is most probable cause, manual cleanup required!"
                        )
                        updated = True
        if len(target_existing) > 0:
            self.log.warning("Found files in target that are not in library:")
            updated = True
            for file in target_existing:
                if delete is True and dry_run is False:
                    os.remove(file)
                    self.log.warning(f"Deleted {file}")
                else:
                    self.log.warning(f"Would delete {file}")
        if updated:
            self.log.info(
                f"Updated, new: {new_docs}, updated: {upd_docs}, debris: {len(target_existing)}"
            )
            if len(upd_doc_names) > 0:
                self.log.info("Updated files:")
                for doc in upd_doc_names:
                    self.log.info(doc)
        else:
            self.log.info("No updates and no new files found, nothing to do.")

    def save_state(self, target_path):
        repo_state = os.path.join(target_path, "repo_state.json")
        with open(repo_state, "w") as f:
            json.dump(self.lib_entries, f, indent=4)

    def load_state(self, target_path):
        repo_state = os.path.join(target_path, "repo_state.json")
        if not os.path.exists(repo_state):
            self.log.error(f"State file not found at {repo_state}")
            return
        with open(repo_state, "r") as f:
            self.lib_entries = json.load(f)
        return self.lib_entries

    @staticmethod
    def _gen_thumbnail(image_path, thumb_dir, thumb_dir_full, uuid, size=(128, 128)):
        dest_path_full = os.path.join(thumb_dir_full, uuid + ".jpg")
        dest_path_rel = os.path.join(thumb_dir, uuid + ".jpg")

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
        self, output_path, max_entries=None, cover_rel_path=None
    ):
        output_path = os.path.expanduser(output_path)
        if not os.path.exists(output_path):
            self.log.info(f"Creating output path {output_path}")
            os.makedirs(output_path)
        if cover_rel_path is None:
            cover_rel_path = "Covers"
        cover_full_path = os.path.join(output_path, cover_rel_path)
        if not os.path.exists(cover_full_path):
            os.makedirs(cover_full_path)
        n = 0
        errs = 0
        for entry in self.lib_entries:
            mandatory_fields = [
                "title",
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
            md = f"---\ncreation: {entry['date_added']}\ntitle: {entry['title']}\nuuid: {entry['uuid']}\nauthors:\n"
            # foot_tags=''
            foot_authors = ""
            authors = entry["creators"]
            for author in authors:
                md += f"  - {author}\n"
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
            for field in entry.keys():
                if (
                    field not in mandatory_fields
                    and field not in special_fields
                    and entry[field] is not None
                ):
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
                cover_path = CalibreTools._gen_thumbnail(
                    entry["cover"], cover_rel_path, cover_full_path, entry["uuid"]
                )
                md += f"![{sanitized_title}]({cover_path})\n\n"
            if "description" in entry.keys() and entry["description"] is not None:
                cmt = entry["description"].replace("\n", "\n> ")
                md += f"> {cmt}\n"
            # if len(foot_tags) > 3:
            #     md += f"\nTags: {foot_tags[:-2]}\n"
            if len(foot_authors) > 3:
                md += f"\nAuthors: {foot_authors}\n"
            if os.path.exists(md_filename):
                print(f"File {md_filename} already exists")
                errs += 1
                continue
            with open(md_filename, "w") as f:
                f.write(md)
            n += 1
            if n == max_entries:
                break
        return n, errs
