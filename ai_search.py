import logging
import os
import json
import time
from typing import TypedDict, override
import ollama
import numpy.typing
import numpy as np


class EmbeddingEntryOld(TypedDict):
    filename: str
    text: str
    embedding_generator: str
    embedding: numpy.typing.NDArray[np.float32]

    
class EmbeddingEntry(TypedDict):
    filename: str
    text: str
    emb_ten_idx: int
    emb_ten_size: int


class EmbeddingSearch:
    def __init__(self, embeddings_path: str, epsilon: float = 1e-6):
        self.log: logging.Logger = logging.getLogger("EmbSearch")
        self.modes: list[str] = ["filepath", "textlibrary"]
        self.epsilon: float = epsilon
        self.texts: dict[str, EmbeddingEntry] = {}
        self.emb_ten: np.typing.NDArray[np.float32] | None = None
        self.repos: dict[str, str] = {}
        e_path = os.path.expanduser(embeddings_path)
        if os.path.exists(e_path) is False:
            self.log.error(f"embeddings_path={embeddings_path} does not exist!")
        self.embeddings_path: str = e_path
        _ = self.load_text_embeddings()
        _ = self.load_repos()
        # Disable verbose Ollama:
        murks_logger = logging.getLogger("httpx")
        murks_logger.setLevel(logging.ERROR)

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
                            'emb_ten_idx': -1,
                            'emb_ten_size': -1
                        }
                        if descriptor_path in self.texts:
                            if self.texts[descriptor_path]['text'] == doc_text:
                                continue
                            else:
                                self.log.info(f"Document {descriptor_path} has been modified, recalculating")
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
        if self.emb_ten is None:
            self.log.error("No embeddings available")
            return
        emb_file = os.path.join(self.embeddings_path, f"library_embeddings.npz")
        desc_file = os.path.join(self.embeddings_path, f"library_desc.json")
        np.savez(emb_file, array=self.emb_ten)
        with open(desc_file, 'w') as f:
            json.dump(self.texts, f)
        self.log.info(f"Info saved {emb_file} and {desc_file}")

    def load_text_embeddings(self) -> int:
        # Start migration
        emb_file_old = os.path.join(self.embeddings_path, 'library_embeddings.json')
        emb_file = os.path.join(self.embeddings_path, f"library_embeddings.npz")
        desc_file = os.path.join(self.embeddings_path, f"library_desc.json")
        if os.path.exists(emb_file) is False and os.path.exists(emb_file_old) is True:
            self.log.warning(f"Starting migration of old {emb_file_old}")
            old_texts: dict[str, EmbeddingEntryOld] | None = None
            with open(emb_file_old, 'r') as f:
                old_texts  = json.load(f)
            if old_texts is None:
                self.log.error("No texts to migrate!")
                return 0
            index = 0
            for txt_desc in old_texts:
                #emb: numpy.typing.ArrayLike | None = None
                # for key in old_texts[txt_desc]:
                emb = np.asarray(old_texts[txt_desc]['embedding'], dtype=np.float32)
                if emb.shape[1] != 768:  # XXX model dep.
                    self.log.error(f"Embedding for {txt_desc} is invalid: {emb.shape}")
                    continue
                for sub_index in range(emb.shape[0]):  # Normalize once
                    embi: np.typing.NDArray[np.float32] = emb[sub_index]
                    embin = embi / np.linalg.norm(embi)
                    emb[sub_index] = embin
                if self.emb_ten is None:
                    self.emb_ten = np.asarray(emb, dtype=np.float32)
                    self.log.info(f"Created emb_ten as {self.emb_ten.shape}")
                else:
                    self.emb_ten = np.append(self.emb_ten, emb, axis=0)
                    if index < 10000:
                        self.log.info(f"index: {index}, emb_ten: {self.emb_ten.shape}, add {emb.shape}")
                self.texts[txt_desc] = {
                    'filename': old_texts[txt_desc]['filename'],
                    'text': old_texts[txt_desc]['text'],
                    'emb_ten_idx': index,
                    'emb_ten_size': emb.shape[0]
                    }
                index += emb.shape[0]
            self.save_text_embeddings()
            if self.emb_ten is not None:
                self.log.warning(f"Migration done: emb_ten: {self.emb_ten.shape}.")
            else:
                self.log.error("Migration failed, no emb_ten")
        else:
        # End migration
            if os.path.exists(emb_file):
                self.emb_ten = np.load(emb_file)['array']
                with open(desc_file, 'r') as f:
                    self.texts  = json.load(f)
        count = len(self.texts)
        self.log.info("Embeddings loaded: texts: {len(self.texts)}, emb_ten: {self.emb_ten.shape}")
        return count
                
    def gen_embeddings(self, model: str, library_name: str, verbose: bool=False, chunk_size: int=2048, save_every_sec: float | None=60):
        lib_desc = '{' + library_name + '}'
        last_save = time.time()
        cnt: int = 0
        max_cnt = len(self.texts.keys())
        index = 0
        for desc in self.texts:
            if self.texts[desc]['emb_ten_idx'] != -1 and self.texts[desc]['emb_ten_size'] != -1:
                index = self.texts[desc]['emb_ten_idx'] + self.texts[desc]['emb_ten_size']
                cnt += 1
                self.log.info(f"Skipping {desc}, already processed, {cnt}/{max_cnt}, index={index}")
                continue
            if desc.startswith(lib_desc):
                text: str = self.texts[desc]['text']
                if len(text) == 0:
                    self.log.warning(f"Text for {desc} is empty, ignoring!")
                    continue
                text_chunks = [self.get_chunk(text, i) for i in range(len(text) // chunk_size)]
                # embeddings: np.ndarray([], dtype=np.float32)
                self.texts[desc]['emb_ten_idx'] = index
                self.texts[desc]['emb_ten_size'] = len(text_chunks)
                response = ollama.embed(model=model, input=text_chunks)
                if len(response['embeddings']) != len(text_chunks):
                    self.log.error(f"Assumption on ollama API failed, can't generate embeddings for {desc}, embs: {len(response['embeddings'])}, chunks: {len(text_chunks)}")
                    continue
                embedding = np.asarray(response["embeddings"], dtype=np.float32)
                if len(embedding.shape)<2 or embedding.shape[1]!=768 or embedding.shape[0] != len(text_chunks):
                    self.log.error(f"Assumption on numpy conversion failed, can't generate embeddings for {desc}, result: {embedding.shape}, chunks: {len(text_chunks)}")
                    continue
                if self.emb_ten is None:
                    self.emb_ten = embedding
                else:
                    self.emb_ten = np.append(self.emb_ten, embedding, axis=0)  
                # for text_chunk in text_chunks:
                #     response = ollama.embed(model=model, input=text_chunk)
                #     embedding = np.asarray(np.array(response["embeddings"][0], dtype=np.float32), dtype=np.float32)
                #     # Check for ignored parts > [0] ! XXX
                #     if len(response["embeddings"]) > 1:
                #         self.log.error(f"Embedding vector has additional components: {len(response['embeddings'])}")
                #     if self.emb_ten is None:
                #         self.emb_ten = np.asarray([embedding], dtype=np.float32)
                #     else:
                #         self.emb_ten = np.append(self.emb_ten, np.array([embedding], dtype=np.float32), axis=0)
                index += len(text_chunks)
                cnt += 1
                if verbose is True and self.emb_ten is not None:
                     print(f"Generated {cnt}/{max_cnt}: {self.emb_ten.shape}, embeddings for {desc}")
                if save_every_sec is not None:
                    if time.time() - last_save > save_every_sec:
                        self.save_text_embeddings()
                        last_save = time.time()
        self.save_text_embeddings()

    # Unused, since everything is normalized
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
        search_embedding = search_embedding / np.linalg.norm(search_embedding)
        if verbose is True:
            self.log.info(f"Search-embedding: {search_embedding.shape}")
        best_doc: str = ""
        best_index: int = -1
        best_chunk: str = ""
        cos_val: float = 0.0
        if self.emb_ten is None or self.texts == {}:
            return best_doc, best_index, best_chunk, cos_val
        t0 = time.time()
        idx_vec = np.asarray(np.matmul(self.emb_ten, search_embedding), dtype=np.float32)
        arg_max = np.argmax(idx_vec)
        for desc in self.texts:  # XXX binary search
            entry = self.texts[desc]
            if entry['emb_ten_idx'] <= arg_max and entry['emb_ten_idx']+entry['emb_ten_size'] > arg_max:
                self.log.info(desc)
                best_doc = desc
                best_index = int(arg_max) - entry['emb_ten_idx']
                cos_val = idx_vec[arg_max]
                best_text = entry['text']
                best_chunk = self.get_chunk(best_text, best_index).replace('\n',' ')
                break
        dt = time.time() - t0
        print(f"Search-time (dim: {self.emb_ten.shape}): {dt:.4f} sec")

        print(best_chunk)
#        self.log.info(f"idx_vec: {idx_vec.shape}, {arg_max}")
#        for desc in self.texts:
#            entry =  self.texts[desc]
#            embeddings = self.emb_ten[entry['emb_ten_idx']:entry['emb_ten_idx']+entry['emb_ten_size'], :]
#            for index in range(embeddings.shape[0]):
#                chunk = embeddings[index,:]
#                if search_embedding.shape != chunk.shape:
#                    self.log.error(f"{entry['filename']}: Invalid chunk.shape {chunk.shape} instead emb shape {search_embedding.shape}, can't compare at index {index}/{embeddings.shape[0]}!")
#                    continue
#                cs: float = np.dot(search_embedding, chunk)  # self.cos_sim(search_embedding, chunk)
#                if cs > cos_val:
#                    best_doc = desc
#                    best_index = index
#                    best_idx = entry['emb_ten_idx']
#                    cos_val = cs
#                    best_text = entry['text']
#                    best_chunk = self.get_chunk(best_text, index).replace('\n',' ')
#                    if verbose is True:
#                        print("\n----\n")
#                        print(f"Best match {cs}: {best_doc}[{entry['emb_ten_idx']}+{index}]: {best_chunk}")
#        if verbose is True:
#            print("------------------")
#            print(f"{search_text}: {best_chunk}, {best_index}")
#        self.log.info(f"idx: {best_index}+{best_idx}")
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
