import logging
import os
import json
import time
from typing import TypedDict, override, cast
import numpy.typing
import numpy as np
import pymupdf  # pyright: ignore[reportMissingTypeStubs]
try:
    import ollama
    ollama_available:bool = True
except ImportError:
    ollama_available = False


class OllamaEmbeddings:
    def __init__(self, model:str, model_location:str="", matmul:str="numpy"):
        self.log: logging.Logger = logging.getLogger("OllamaEmbedder")
        self.model_available:bool = ollama_available
        if ollama_available is False:
            self.log.error("Failed to load ollama module, is it installed?")
        matmuls = ["numpy"]
        if matmul in matmuls:
            self.matmul_engine:str = matmul
        else:
            self.matmul_engine = "numpy"
        if ollama_available is True:
            try:
                _ = ollama.show(model)  # pyright:ignore[reportPossiblyUnboundVariable]
            except Exception as e:
                self.log.error(f"Failed to load model {model} with ollama: {e}")
                self.model_available = False
            self.model: str = model
            # Disable verbose Ollama:
            murks_logger = logging.getLogger("httpx")
            murks_logger.setLevel(logging.ERROR)
    
    def embed(self, text_chunks: list[str], description:str | None=None, normalize:bool=True) -> np.typing.NDArray[np.float32]:
        if self.model_available is False:
            self.log.error("Embeddings model is not available")
            return np.asarray([], dtype=np.float32)
        if description is not None:
            self.log.info(f"Generating embedding for {description}...")
        response = ollama.embed(model=self.model, input=text_chunks)  # pyright:ignore[reportPossiblyUnboundVariable]
        embeddings: np.typing.NDArray[np.float32] = np.asarray(response["embeddings"], dtype=np.float32)
        if normalize is True:
            embeddings = (embeddings.transpose() / np.linalg.norm(embeddings, axis=1)).transpose()  #pyright:ignore[reportAny]
        return embeddings

    def matmul(self, embeddings:np.typing.NDArray[np.float32], search_vector:np.typing.NDArray[np.float32]) -> np.typing.NDArray[np.float32]:
        if self.matmul_engine == "numpy":
            if embeddings.shape[1] != search_vector.shape[0]:
                self.log.error(f"Shape mismatch on matmul: embeddings[{embeddings.shape}] x search_vection[{search_vector.shape[0]}], because {embeddings.shape[1]} != {search_vector.shape[0]}")
                return np.asarray([], dtype=np.float32)
            mul = np.asarray(np.matmul(embeddings, search_vector), dtype=np.float32)
        else:
            self.log.error("Matmul engine {self.matmul_engine} not implemented!")
            return np.asarray([], dtype=np.float32)
        return mul


class EmbeddingEntry(TypedDict):
    filename: str
    text: str
    page_no: int
    emb_ten_idx: int
    emb_ten_size: int

class SearchResult(TypedDict):
    cosine: float
    index: int
    desc: str
    text: str
    chunk: str
    yellow_liner: np.typing.NDArray[np.float32] | None


class EmbeddingsSearch:
    def __init__(self, embeddings_path: str, model: str, version:str="", embeddings_engine:str = "ollama", epsilon: float = 1e-6):
        self.log: logging.Logger = logging.getLogger("EmbSearch")
        self.modes: list[str] = ["filepath", "textlibrary"]
        self.epsilon: float = epsilon
        self.model:str = model
        self.version:str = version
        embeddings_engines: list[str] = ["ollama"]
        if embeddings_engine in embeddings_engines:
            self.embeddings_engine:str = embeddings_engine
            if self.embeddings_engine == "ollama":
                self.engine: OllamaEmbeddings = OllamaEmbeddings(self.model)
            
        self.texts: dict[str, EmbeddingEntry] = {}
        self.emb_ten: np.typing.NDArray[np.float32] | None = None
        self.texts_ptr_list: list[str] = []
        self.emb_ten_idx_to_text_ptr: list[int] = []
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

    def read_pdf_library(self, library_name: str, library_path: str, pdf_cache:str) -> int:
        l_path = os.path.abspath(os.path.expanduser(library_path))
        count = 0
        if os.path.exists(l_path) is False:
            self.log.error("library_path {library_path} does not exist!")
            return count
        if library_name in self.repos and self.repos[library_name] != l_path and os.path.abspath(os.path.expanduser(self.repos[library_name])) != l_path:
            self.log.error(f"libray_name {library_name} already registered with different path {l_path} != {self.repos[library_name]} or {os.path.abspath(os.path.expanduser(self.repos[library_name]))}, ignored!")
        else:
            home = os.path.expanduser("~")
            if library_path.startswith(home):
                library_path = "~" + library_path[len(home):]
            self.repos[library_name] = library_path
            self.save_repos()
        for root, _dir, files in os.walk(l_path):
            for file in files:
                if file.endswith('.pdf'):
                    rel_path = root[len(l_path)+1:]
                    
                    pdf_cache_path = os.path.join(pdf_cache, rel_path)
                    cache_file = os.path.splitext(file)[0] + ".txt"
                    pdf_cache_file = os.path.join(pdf_cache_path, cache_file)
                    descriptor_path = "{" + library_name + "}" +f"{rel_path}/{file}"
                    if os.path.exists(pdf_cache_file) is True:
                        with open(pdf_cache_file, "r") as f:
                            pdf_text = f.read()
                        page_no = 1
                    else:
                        full_path = os.path.join(root, file)
                        doc = pymupdf.open(full_path)
                        page_no:int = 0
                        entry: EmbeddingEntry
                        pdf_text: str = ""
                        for page in doc:
                            page_no += 1
                            page_text = page.get_text()  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue, reportUnknownVariableType]
                            if isinstance(page_text, str) is False:
                                self.log.error(f"Can't read {rel_path} page {page_no}, ignoring page")
                                continue
                            page_text = cast(str, page_text)
                            if len(page_text) == 0:
                                continue
                            pdf_text += page_text
                        os.makedirs(pdf_cache_path, exist_ok=True)
                        with open(pdf_cache_file, 'w') as f:
                            _ = f.write(pdf_text)
                    entry = {
                        'filename': file,
                        'text': pdf_text,
                        'page_no': page_no,
                        'emb_ten_idx': -1,
                        'emb_ten_size': -1
                    }
                    if descriptor_path in self.texts:
                        if self.texts[descriptor_path]['text'] == pdf_text:
                            continue
                    self.texts[descriptor_path] = entry
                    count += 1                       
        return count

    def read_text_library(self, library_name: str, library_path: str, extensions:list[str] | None = None) -> int:
        if extensions is None:
            extensions = [".txt"]
        l_path = os.path.abspath(os.path.expanduser(library_path))
        count = 0
        if os.path.exists(l_path) is False:
            self.log.error("library_path {library_path} does not exist!")
            return count
        if library_name in self.repos and self.repos[library_name] != l_path and os.path.abspath(os.path.expanduser(self.repos[library_name])) != l_path:
            self.log.error(f"libray_name {library_name} already registered with different path {l_path} != {self.repos[library_name]} or {os.path.abspath(os.path.expanduser(self.repos[library_name]))}, ignored!")
        else:
            home = os.path.expanduser("~")
            if library_path.startswith(home):
                library_path = "~" + library_path[len(home):]
            self.repos[library_name] = library_path
            self.save_repos()
        for root, _dir, files in os.walk(l_path):
            for file in files:
                ext = os.path.splitext(file)
                if len(ext)==2:
                    ext = ext[1]
                else:
                    self.log.error(f"Can't identify extension of {file}, ignoring")
                    continue
                if ext in extensions:
                    rel_path = root[len(l_path)+1:]
                    full_path = os.path.join(root, file)
                    with open(full_path, 'r', encoding='utf-8') as f:
                        try:
                            doc_text = f.read()
                        except Exception as e:
                            self.log.error(f"Failed to read {full_path}, {e}")
                            continue
                        descriptor_path = "{" + library_name + "}" +f"{rel_path}/{file}"
                        entry: EmbeddingEntry = {
                            'filename': file,
                            'text': doc_text,
                            'page_no': -1,
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

    def get_chunk_cut(self, text: str, index: int, av_chunk_size: int=2048, av_chunk_overlay:int=0) -> str:
        chunk = text[index*(av_chunk_size-av_chunk_overlay):(index+1)*(av_chunk_size-av_chunk_overlay)]
        return chunk

    def get_chunk(self, text: str, index: int, av_chunk_size: int=2048, av_chunk_overlay:int=0, chunk_method:str = "cut") -> str:
        if chunk_method == "cut":
            return self.get_chunk_cut(text, index, av_chunk_size, av_chunk_overlay);
        else:
            self.log.error("Invalid chunk_method {chunk_method}")
            return ""

    def get_chunks(self, text:str, av_chunk_size:int=2048, av_chunk_overlay:int=0, chunk_method:str="cut") -> list[str]:
        if chunk_method == "cut":
            chunks = (len(text) - 1) // (av_chunk_size - av_chunk_overlay) + 1 
            text_chunks = [self.get_chunk_cut(text, i) for i in range(chunks) ]
            return text_chunks
        else:
            self.log.error(f"Unknown chunk_method: {chunk_method}")
            return []
            

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
        emb_file = os.path.join(self.embeddings_path, f"{self.model}{self.version}_library_embeddings.npz")
        desc_file = os.path.join(self.embeddings_path, f"{self.model}{self.version}_library_desc.json")
        np.savez(emb_file, array=self.emb_ten)
        with open(desc_file, 'w') as f:
            json.dump(self.texts, f)
        self.log.info(f"Info saved {emb_file} and {desc_file}")

    def load_text_embeddings(self, normalize:bool = False) -> int:
        migrate:bool = False
        emb_file = os.path.join(self.embeddings_path, f"{self.model}{self.version}_library_embeddings.npz")
        if os.path.exists(emb_file) is False:
            emb_file = os.path.join(self.embeddings_path, f"library_embeddings.npz")
            if os.path.exists(emb_file) is True:
                self.log.warning(f"Migrating old embeddings filename {emb_file}")
                migrate = True
        desc_file = os.path.join(self.embeddings_path, f"{self.model}{self.version}_library_desc.json")
        if os.path.exists(desc_file) is False:
            desc_file = os.path.join(self.embeddings_path, f"library_desc.json")
            if os.path.exists(desc_file) is True:
                self.log.warning(f"Migrating old embeddings description filename {desc_file}")
                migrate = True
        if os.path.exists(emb_file):
            self.emb_ten = np.load(emb_file)['array']
            if normalize is True and self.emb_ten is not None:
                self.emb_ten = self.emb_ten / np.linalg.norm(self.emb_ten, axis=1)
                self.log.warning("NORMALIZED on LOAD! OBSOLETE CODE!")
            with open(desc_file, 'r') as f:
                self.texts  = json.load(f)
        count = len(self.texts)
        if self.emb_ten is not None:
            self.log.info(f"Embeddings loaded: texts: {len(self.texts)}, emb_ten: {self.emb_ten.shape}")
        else:
            self.log.info(f"Embeddings loaded: texts: {len(self.texts)}")
        if migrate is True:
            self.save_text_embeddings()
            self.log.info("Migration of embeddings-files complete.")
        self.texts_ptr_list = list(self.texts.keys())
        tpi = 0
        eti: int = self.texts[self.texts_ptr_list[tpi]]['emb_ten_idx']
        ets: int = self.texts[self.texts_ptr_list[tpi]]['emb_ten_size']
        if self.emb_ten is not None:
            self.emb_ten_idx_to_text_ptr = []
            for i in range(self.emb_ten.shape[0]):
                if i<eti:
                    self.log.error(f"Alg failed (1) at {i}, {eti}")
                    exit(1)
                if i>eti+ets:
                    self.log.error(f"Alg failed (2) at {i}, {ets}, {eti}")
                    exit(1)
                self.emb_ten_idx_to_text_ptr.append(tpi)
                if i == eti+ets:
                    eti = -1
                    while eti == -1:
                        tpi += 1
                        eti = self.texts[self.texts_ptr_list[tpi]]['emb_ten_idx']
                        ets = self.texts[self.texts_ptr_list[tpi]]['emb_ten_size']
            if len(self.emb_ten_idx_to_text_ptr) != self.emb_ten.shape[0]:
                self.log.error(f"Alg failure (3) {len(self.emb_ten_idx_to_text_ptr)} != {self.emb_ten.shape[0]}")
                exit(1)
        return count
                
    def gen_embeddings(self, library_name: str, verbose: bool=False, av_chunk_size: int=2048, av_chunk_overlay: int=0,
                       chunk_method:str = "cut",  save_every_sec: float | None=360):
        lib_desc = '{' + library_name + '}'
        last_save = time.time()
        cnt: int = 0
        max_cnt = len(self.texts.keys())
        if self.emb_ten is None:
            index = 0
        else:
            index = len(self.emb_ten)
        for desc in self.texts:
            if self.texts[desc]['emb_ten_idx'] != -1 and self.texts[desc]['emb_ten_size'] != -1:
                cnt += 1
                continue
            if desc.startswith(lib_desc):
                text: str = self.texts[desc]['text']
                if len(text) == 0:
                    continue
                text_chunks = self.get_chunks(text, av_chunk_size, av_chunk_overlay, chunk_method)
                self.texts[desc]['emb_ten_idx'] = index
                self.texts[desc]['emb_ten_size'] = len(text_chunks)

                t_start = time.time()
                embeddings = self.engine.embed(text_chunks, desc, normalize=True)
                
                if len(embeddings.shape)<2 or embeddings.shape[0] != len(text_chunks):
                    self.log.error(f"Assumption on numpy conversion failed, can't generate embeddings for {desc}, result: {embeddings.shape}, chunks: {len(text_chunks)}")
                    continue
                if self.emb_ten is None:
                    self.emb_ten = embeddings
                else:
                    self.emb_ten = np.append(self.emb_ten, embeddings, axis=0)  
                index = len(self.emb_ten)
                cnt += 1
                dt = (time.time() - t_start) / len(text_chunks)
                if verbose is True and self.emb_ten is not None:
                     print(f"   Generated {cnt}/{max_cnt}: {self.emb_ten.shape}, {dt:0.5f} sec per embedding-vector")
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

    def yellow_line_it(self, text: str, search_embeddings: np.typing.NDArray[np.float32], context:int=16, context_steps:int=1) -> np.typing.NDArray[np.float32]:
        clr: list[str] = []
        for i in range(0, len(text), context_steps):
            i0 = i - context // 2
            i1 = i + context // 2
            if i0 < 0:
                i1 = i1 - i0
                i0 = 0
            if i1 > len(text):
                i0 = i0 - (i1 - len(text))
                i1 = len(text)
            clr.append(text[i0:i1])
        if clr == []:
            clr = [text]
        embs = self.engine.embed(clr, normalize=True)
        yellow = np.asarray(self.engine.matmul(embs, search_embeddings.transpose()), np.float32)
        return yellow

    def search_embeddings(self, search_text: str, verbose: bool=False, av_chunk_size: int=2048, av_chunk_overlay: int=0,
                          chunk_method:str="cut", yellow_liner: bool=False, context:int=16, context_steps:int=1, max_results:int=10) -> list[SearchResult] | None:

        search_text_list = [search_text]
        search_embeddings = self.engine.embed(search_text_list, normalize=True)
        result: SearchResult
        if len(search_text) > av_chunk_size:
            self.log.warning(f"Search text is longer than av_chunk_size: {len(search_text)}")
        if verbose is True:
            self.log.info(f"Search-embedding: {search_embeddings.shape}")
        results: list[SearchResult] = []
        
        if self.emb_ten is None or self.texts == {}:
            return None
        t0 = time.time()

        t0 = time.time()
        idx_vec: np.typing.ArrayLike = self.engine.matmul(self.emb_ten, search_embeddings.transpose())
        idx_list = cast(list[float], idx_vec.tolist())
        idx_idx = list(enumerate(idx_list))
        idx_srt = sorted(idx_idx, key=lambda x: x[1], reverse=True)
        dt = time.time() - t0
        if dt > 0:
            dn = self.emb_ten.shape[0] / dt
            self.log.info(f"Search: {dt:0.4f} sec, performance: {dn:0.2f} embeddings per second compared")
        for res_i in range(max_results):
            arg_max_i = idx_srt[res_i][0]
            ptr = self.emb_ten_idx_to_text_ptr[arg_max_i]
            txt_name: str = self.texts_ptr_list[ptr]
            entry = self.texts[txt_name] 
            self.log.info(f"{res_i}. {idx_vec[arg_max_i]}: {txt_name}")
            index = arg_max_i - entry['emb_ten_idx']
            chunk = self.get_chunk(entry['text'], index, av_chunk_size, av_chunk_overlay, chunk_method)
            if yellow_liner is True:
                yellow_liner_weights = self.yellow_line_it(chunk, search_embeddings, context=context, context_steps=context_steps)
            else:
                yellow_liner_weights = None
            if yellow_liner_weights is not None:
                result = {
                    'cosine': idx_vec[arg_max_i][0],
                    'desc': txt_name,
                    'index': index,
                    'text': entry['text'],
                    'chunk': chunk.replace('\n',' '),
                    'yellow_liner': yellow_liner_weights[:,0]
                    }
                results.append(result)
            else:
                result = {
                    'cosine': idx_vec[arg_max_i][0],
                    'desc': txt_name,
                    'index': index,
                    'text': entry['text'],
                    'chunk': chunk.replace('\n',' '),
                    'yellow_liner': None
                    }
                results.append(result)
                
        results = sorted(results, key=lambda x: x['cosine'], reverse=True)
        dt = time.time() - t0
        if verbose is True:
            self.log.info(f"Search-time (dim: {self.emb_ten.shape}): {dt:.4f} sec")
        return results
    

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
