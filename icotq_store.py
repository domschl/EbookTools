import logging
import os
import json
from typing import TypedDict


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


class IcoTqStore:
    def __init__(self) -> None:
        self.log:logging.Logger = logging.getLogger("IcoTqStore")
        config_path = os.path.expanduser("~/.config/icotq")  # Turquoise icosaeder
        if os.path.isdir(config_path) is False:
            os.makedirs(config_path)
        self.lib: list[LibEntry] = []
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
                    'file_types': ['txt', 'pdf', 'epub']
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
        lib_path = os.path.join(self.root_path, "icotq_library.json")
        if os.path.exists(lib_path) is True:
            with open(lib_path, 'r') as f:
                self.lib = json.load(f)
        else:
            self.log.info(f"No current library state at {lib_path}")

    def write_library(self):
        lib_path = os.path.join(self.root_path, "icotq_library.json")
        with open(lib_path, 'w') as f:
            json.dump(self.lib, f)

    def import_texts(self):
        if len(self.config['tq_sources']) == 0:
            self.log.error(f"No valid sources defined in config, can't import")
            return
        lib_changed = False
        for source in self.config['tq_sources']:
            source_path = os.path.expanduser(source['path'])
            for root, _dir, files in os.walk(source_path):
                for file in files:
                    ext = os.path.splitext(file)[1]
                    if len(ext) > 0:
                        ext = ext[1:]  # remove leading '.'
                    if ext not in source['file_types']:
                        continue
                    full_path = os.path.join(root, file)
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
                            print("can't do PDF yet")
                            continue
                        else:
                            self.log.error(f"Unsupported converstion {ext} to text")
                            continue
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

            

    