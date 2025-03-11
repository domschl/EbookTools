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


class IcoTqStore:
    def __init__(self) -> None:
        self.log:logging.Logger = logging.getLogger("IcoTqStore")
        config_path = os.path.expanduser("~/.config/icotq")  # Turquoise icosaeder
        if os.path.isdir(config_path) is False:
            os.makedirs(config_path)
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
        root_path = os.path.expanduser(self.config['icotq_path'])
        if os.path.exists(root_path) is False:
            os.makedirs(root_path)
            self.log.warning(f"Creating IcoTq storage path at {root_path}, all IcoTq data will reside there. Modify {config_file} to chose another location!")
        config_subdirs = ['Texts', 'Embeddings', 'PDFTextCache', 'EpubTextCache']
        for cdir in config_subdirs:
            full_path = os.path.join(root_path, cdir)
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
        
    def list_sources(self) -> None:
        for id, source in enumerate(self.config['tq_sources']):
            print(f"{id:02d} {source['name']}, {source['tqtype']}, {source['path']}, {source['file_types']}")
