import logging
import os
import json
import uuid
from typing import TypedDict, cast
import pymupdf  # pyright: ignore[reportMissingTypeStubs]


class TqSource(TypedDict):
    name: str
    tqtype: str
    path: str
    file_types: list[str]


class IcotqConfig(TypedDict):
    icotq_path: str
    tq_sources: list[TqSource]
    ebook_mirror: str


class LibEntry(TypedDict):
    source_name: str
    filename: str
    desc_filename: str
    text: str


class PDFIndex(TypedDict):
    previous_failure: bool
    filename: str
    file_size: int


class IcoTqStore:
    def __init__(self) -> None:
        self.log:logging.Logger = logging.getLogger("IcoTqStore")
        config_path = os.path.expanduser("~/.config/icotq")  # Turquoise icosaeder
        if os.path.isdir(config_path) is False:
            os.makedirs(config_path)
        self.lib: list[LibEntry] = []
        self.pdf_index:dict[str, PDFIndex] = {}
        config_file = os.path.join(config_path, "icoqt.json")
        self.config:IcotqConfig
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                iqc: IcotqConfig = json.load(f)
                self.config = iqc
        else:
            self.config = IcotqConfig({
                'icotq_path': '~/IcoTqStore',
                'tq_sources': [
                    TqSource({
                    'name': 'Calibre',
                    'tqtype': 'calibre_library',
                    'path': '~/ReferenceLibrary/Calibre Library',
                    'file_types': ['txt', 'pdf']
                    }),
                    TqSource({
                        'name': 'Notes',
                        'tqtype': 'folder',
                        'path': '~/Notes',
                        'file_types': ['md']
                    })],
                'ebook_mirror': '~/MetaLibrary'
                })
            with open(config_file, 'w') as f:
                json.dump(self.config,f)
            self.log.warning(f"Created default configuration at {config_file}, please review!")
        self.root_path:str = os.path.expanduser(self.config['icotq_path'])
        if os.path.exists(self.root_path) is False:
            os.makedirs(self.root_path)
            self.log.warning(f"Creating IcoTq storage path at {self.root_path}, all IcoTq data will reside there. Modify {config_file} to chose another location!")
        config_subdirs = ['Texts', 'Embeddings', 'PDFTextCache', 'EpubTextCache']
        for cdir in config_subdirs:
            full_path = os.path.join(self.root_path, cdir)
            if os.path.isdir(full_path) is False:
                os.makedirs(full_path)
        for source in self.config['tq_sources']:
            valid:bool = True
            known_types: list[str] = ['txt', 'epub', 'md', 'pdf']
            for tp in source['file_types']:
                if tp not in known_types:
                    self.log.error(f"Source {source} has invalid file type {tp}, allowed are {known_types}, ignoring this source!")
                    valid = False
                    break
            if os.path.exists(os.path.expanduser(source['path'])) is False:
                self.log.error(f"Source {source} has invalid file path {source['path']}, ignoring this source!")
                valid = False
            known_tqtypes = ['calibre_library', 'folder']
            if source['tqtype'] not in known_tqtypes:
                self.log.error(f"Source {source} has invalid tqtype {source['tqtype']}, valid are {known_tqtypes}, ignoring this source!")
                valid = False
            if valid is False:
                self.config['tq_sources'].remove(source)
                self.log.warning(f"Please fix configuration file {config_file}")
        self.read_library()
        
    def list_sources(self) -> None:
        for id, source in enumerate(self.config['tq_sources']):
            print(f"{id:02d} {source['name']}, {source['tqtype']}, {source['path']}, {source['file_types']}")

    def read_library(self):
        print("\rLoading library...", end="", flush=True)
        lib_path = os.path.join(self.root_path, "icotq_library.json")
        if os.path.exists(lib_path) is True:
            with open(lib_path, 'r') as f:
                self.lib = json.load(f)
            print("\r", end="", flush=True)
            self.log.info(f"Library loaded, {len(self.lib)} entries")
        else:
            print("\r", end="", flush=True)
            self.log.info(f"No current library state at {lib_path}")
        pdf_cache = os.path.join(self.root_path, "PDFTextCache")
        pdf_cache_index = os.path.join(pdf_cache, "pdf_index.json")
        print("\rLoading PDF cache...", end="", flush=True)
        if os.path.exists(pdf_cache_index):
            with open(pdf_cache_index, 'r') as f:
                self.pdf_index = json.load(f)
            print("\r", end="", flush=True)
            self.log.info(f"PDF text cache loaded, {len(self.pdf_index.keys())} entries")
        else:
            self.pdf_index = {}
            print("\r", end="", flush=True)

    def save_pdf_cache_state(self):
        pdf_cache = os.path.join(self.root_path, "PDFTextCache")
        pdf_cache_index = os.path.join(pdf_cache, "pdf_index.json")
        with open(pdf_cache_index, 'w') as f:
            json.dump(self.pdf_index, f)

    def write_library(self):
        lib_path = os.path.join(self.root_path, "icotq_library.json")
        with open(lib_path, 'w') as f:
            json.dump(self.lib, f)
        self.save_pdf_cache_state()

    def get_pdf_text(self, desc:str, full_path:str) -> tuple[str | None, bool]:
        pdf_cache = os.path.join(self.root_path, "PDFTextCache")
        text: str | None = None
        if desc in self.pdf_index:
            cur_file_size = os.path.getsize(full_path)
            if cur_file_size == self.pdf_index[desc]['file_size'] and self.pdf_index[desc]['previous_failure'] is False:
                try:
                    with open(self.pdf_index[desc]['filename'], 'r') as f:
                        text = f.read()
                        return text, False
                except Exception as e:
                    self.log.warning(f"Failed to read PDF cache file for {desc}: {e}")
                    del self.pdf_index[desc]
                    text = None
            else:
                if cur_file_size != self.pdf_index[desc]['file_size']:
                    self.log.info(f"PDF file {full_path} has changed, re-importing")
                    del self.pdf_index[desc]
                else:
                    # self.log.info(f"PDF file {full_path} has no text (extract failed before), ignoring")
                    return None, False  # Known failure case
        changed: bool = False
        if text is None:
            doc = pymupdf.open(full_path)
            text = ""
            for page in doc:
                page_text = page.get_text()  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue, reportUnknownVariableType]
                if isinstance(page_text, str) is False:
                    self.log.error(f"Can't read page of {full_path}, ignoring page")
                    continue
                page_text = cast(str, page_text)
                text += page_text
            if text == "":
                text = None
                failure = True
                cache_filename = ""
                self.log.info(f"Failed to extract text from: {desc}")
            else:
                cache_filename = os.path.join(pdf_cache, str(uuid.uuid4()))
                failure = False
                self.log.info(f"Importing and caching PDF {full_path}")
            pdf_ind: PDFIndex = {
                'filename': cache_filename,
                'file_size': os.path.getsize(full_path),
                'previous_failure': failure
            }
            if failure is False and text is not None:
                with open(pdf_ind['filename'], 'w') as f:
                    _ = f.write(text)
            self.pdf_index[desc] = pdf_ind
            # self.save_pdf_cache_state()
            self.log.info(f"Added {desc} to PDF cache, size: {len(self.pdf_index.keys())}, failure: {failure}")
            changed = True
        return text, changed

    def import_texts(self):
        if len(self.config['tq_sources']) == 0:
            self.log.error(f"No valid sources defined in config, can't import")
            return
        lib_changed = False
        for source in self.config['tq_sources']:
            source_path = os.path.expanduser(source['path'])
            for root, _dir, files in os.walk(source_path):
                for filename in files:
                    parts = os.path.splitext(filename)
                    file_base = parts[0]
                    if len(parts[1]) > 0:
                        ext = parts[1][1:].lower()  # remove leading '.'
                    else:
                        ext = ""
                    if ext not in source['file_types']:
                        continue
                    alt_exists = False
                    if ext in ['epub', 'pdf']:
                        for alt in ['txt', 'epub']:
                            if alt == ext:
                                continue
                            alt_file = os.path.join(root, file_base + '.' + alt)
                            if os.path.exists(alt_file):
                                # self.log.info(f"Better format {alt} exists for {filename}")
                                alt_exists = True
                        if alt_exists is True:  # better format of same file exist, so skip this one
                            continue                    
                    full_path = os.path.join(root, filename)
                    desc_path = "{"+ source['name'] + "}" + full_path[len(source_path):]
                    in_lib = False
                    for entry in self.lib:
                        if entry['desc_filename'] == desc_path:
                            # Check if changed!
                            in_lib = True
                            break
                    if in_lib is False:
                        text = None
                        if ext in ['md', 'py', 'txt']:
                            with open(full_path, 'r') as f:
                                text = f.read()
                        elif ext == 'pdf':
                            text, changed = self.get_pdf_text(desc_path, full_path)
                            if changed is True:
                                lib_changed = True
                        else:
                            self.log.error(f"Unsupported conversion {ext} to text at {desc_path}")
                            continue
                        if text is not None:
                            entry: LibEntry = LibEntry({
                                'source_name': source['name'],
                                'desc_filename': desc_path,
                                'filename': full_path,
                                'text': text
                            })
                            self.lib.append(entry)
                            lib_changed = True
        if lib_changed is True:
            self.write_library()
            self.log.info("Changed library saved.")

            

    