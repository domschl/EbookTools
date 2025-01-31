import logging
import os
import json
import time
from typing import TypedDict, override
import ollama
import numpy.typing
import numpy as np


class EmbeddingEntry(TypedDict):
    filename: str
    text: str
    embedding_generator: str
    embedding: numpy.typing.NDArray[np.float32]

class EmbeddingSearch:
    def __init__(self, embeddings_path: str, epsilon: float = 1e-6):
        self.log: logging.Logger = logging.getLogger("EmbSearch")
        self.modes: list[str] = ["filepath", "textlibrary"]
        self.epsilon: float = epsilon
        self.texts: dict[str, EmbeddingEntry] | None = None
        self.repos: dict[str, str] = {}
        e_path = os.path.expanduser(embeddings_path)
        if os.path.exists(e_path) is False:
            self.log.error(f"embeddings_path={embeddings_path} does not exist!")
        self.embeddings_path: str = e_path
        _ = self.load_text_embeddings()
        _ = self.load_repos()

    def load_repos(self, silent: bool = True) -> int:
        repo_file = os.path.join(self.embeddings_path, 'repos_embeddings.json')
        if os.path.exists(repo_file) is False:
            if silent is False:
                self.log.error(f"Repo file {repo_file} does not exist!")
            return 0
        with open(repo_file, 'r') as f:
            self.repos  = json.load(f)
        return len(self.repos)

    def save_repos(self):
        repo_file = os.path.join(self.embeddings_path, 'repos_embeddings.json')
        with open(repo_file, 'w') as f:
            json.dump(self.repos, f)
        
    def read_text_library(self, library_name: str, library_path: str) -> int: 
        l_path = os.path.abspath(os.path.expanduser(library_path))
        if l_path[-1] != '/':
            l_path += '/'
        count = 0
        if os.path.exists(l_path) is False:
            self.log.error("library_path {library_path} does not exist!")
            return count
        if library_name in self.repos and self.repos[library_name] != l_path:
            self.log.error(f"libray_name {library_name} already registered with different path {l_path} != {self.repos[library_name]}, ignored!")
        else:
            self.repos[library_name] = l_path
            self.save_repos()
        if self.texts is None:
            self.texts = {}
        for root, _dir, files in os.walk(l_path):
            for file in files:
                if file.endswith('.txt'):
                    rel_path = root[len(l_path):]
                    full_path = os.path.join(root, file)
                    with open(full_path, 'r') as f:
                        doc_text = f.read()
                        descriptor_path = "{" + library_name + "}" +f"{rel_path}/{file}"
                        entry: EmbeddingEntry = {
                            'filename': file,
                            'text': doc_text,
                            'embedding_generator': "",
                            'embedding': np.ndarray([])
                        }
                        if descriptor_path in self.texts:
                            if self.texts[descriptor_path]['text'] != doc_text:
                                self.log.error(f"Text changed: {descriptor_path}, resetting all embeddings")
                                self.texts[descriptor_path] = entry
                                count += 1
                        else:
                            self.texts[descriptor_path] = entry
                            count += 1
        return count

    @staticmethod
    def get_chunk(text: str, index: int, chunk_size: int=2048):
        chunk = text[index*chunk_size:(index+1)*chunk_size]
        return chunk

    # This contains many annotations that only serve to shut up type-checker of basedpyright...
    class NumpyEncoder(json.JSONEncoder):
        @override  # Python 3.12 beauty
        def default(self, o):  # pyright: ignore[reportAny, reportMissingParameterType] # type: ignore
            if isinstance(o, np.ndarray):
                return o.tolist()
            return super().default(o)  # pyright: ignore[reportAny] # type: ignore

    def save_text_embeddings(self):
        emb_file = os.path.join(self.embeddings_path, 'library_embeddings.json')
        with open(emb_file, 'w') as f:
            json.dump(self.texts, f, cls=self.NumpyEncoder)

    def load_text_embeddings(self, silent: bool=True) -> int:
        emb_file = os.path.join(self.embeddings_path, 'library_embeddings.json')
        if os.path.exists(emb_file) is False:
            if silent is False:
                self.log.error(f"Embeddings file {emb_file} does not exist!")
            return 0
        with open(emb_file, 'r') as f:
            self.texts  = json.load(f)
        if self.texts is None:
            return 0
        for txt_desc in self.texts:
            for key in self.texts[txt_desc]:
                if key.startswith('emb_'):
                    self.texts[txt_desc][key] = np.asarray(self.texts[txt_desc][key], dtype=np.float32)
        return len(self.texts)
                
    def gen_embeddings(self, model: str, library_name: str, verbose: bool=False, chunk_size: int=2048, save_every_sec: float | None=60):
        lib_desc = '{' + library_name + '}'
        last_save = time.time()
        if self.texts is None:
            return
        for desc in self.texts:
            if desc.startswith(lib_desc):
                if self.texts[desc]['embedding_generator'] == "":
                    text: str = self.texts[desc]['text']
                    text_chunks = [self.get_chunk(text, i) for i in range(len(text) // chunk_size)]
                    # embeddings: np.ndarray([], dtype=np.float32)
                    init: bool = False
                    embeddings = np.array([[]], dtype=np.float32)
                    for text_chunk in text_chunks:
                        response = ollama.embed(model=model, input=text_chunk)
                        embedding: numpy.typing.NDArray[np.float32] = np.array(response["embeddings"][0], dtype=np.float32)
                        if init is False:
                            init = True
                            embeddings: numpy.typing.NDArray[np.float32] = np.array([embedding], dtype=np.float32)
                        else:
                            embeddings = np.append(embeddings, np.array([embedding], dtype=np.float32), axis=0)
                        print(".", end="")
                    print()   
                    self.texts[desc]['embedding'] = embeddings
                    self.texts[desc]['embedding_generator'] = model
                    # del self.texts[desc]['text']
                    if verbose is True:
                         print(f"Generated {self.texts[desc]['embedding'].shape[0]} embeddings for {desc}")
                    if save_every_sec is not None:
                        if time.time() - last_save > save_every_sec:
                            self.save_text_embeddings()
                            last_save = time.time()
        self.save_text_embeddings()

    def cos_sim(self, a: numpy.typing.ArrayLike, b: numpy.typing.ArrayLike) -> float:
        m: float = np.dot(a,b)
        n: float = float(np.linalg.norm(a) * np.linalg.norm(b))
        if n < self.epsilon:
            n = self.epsilon
        return float(m / n)

    def search_embeddings(self, model: str, search_text: str, verbose: bool=False, chunk_size: int=2048) -> tuple[str, int, str, float]:
        search_text = search_text[:chunk_size]
        # while len(search_text) < chunk_size:
        #     search_text = " " + search_text
        response = ollama.embed(model=model, input=search_text)
        search_embedding = np.array(response["embeddings"][0], dtype=np.float32)
        best_doc: str = ""
        best_index: int = -1
        best_chunk: str = ""
        cos_val: float = 0.0
        if self.texts is None:
            return best_doc, best_index, best_chunk, cos_val
        for desc in self.texts:
            entry =  self.texts[desc]
            if entry['embedding'] is None:
                if verbose is True:
                    print(f"No embeddings from model {model} in {entry['filename']}, ignoring doc")
                continue
            for index in range(entry['embedding'].shape[0]):
                chunk = entry['embedding'][index,:]
                cs = self.cos_sim(search_embedding, chunk)
                if cs > cos_val:
                    best_doc = desc
                    best_index = index
                    cos_val = cs
                    best_text = entry['text']
                    best_chunk = self.get_chunk(best_text, index).replace('\n',' ')
                    if verbose is True:
                        print("\n----\n")
                        print(f"Best match {cs}: {best_doc}[{index}]: {best_chunk}")
        if verbose is True:
            print("------------------")
            print(f"{search_text}: {best_chunk}")
        return best_doc, best_index, best_chunk, cos_val


model_list = [
  "nomic-embed-text",
  "mxbai-embed-large",
  "bge-m3",
  "paraphrase-multilingual:278m-mpnet-base-v2-fp16",
  "snowflake-arctic-embed2",
  "granite-embedding:278m",
  "llama3.3:latest",
  "qwen2.5:14b",
  "qwen2.5:7b"
]
