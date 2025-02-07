import logging
import os
import json
import time
from typing import TypedDict, override, cast
import ollama
import numpy.typing
import numpy as np
import pymupdf  # pyright: ignore[reportMissingTypeStubs]


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
    yellow_liner: list[float] | None
    
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

    def read_pdf_library(self, library_name: str, library_path: str, single_pages: bool = False) -> int:
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
                aborted: bool = False
                if file.endswith('.pdf'):
                    rel_path = root[len(l_path):]
                    full_path = os.path.join(root, file)
                    doc = pymupdf.open(full_path)
                    page_no:int = 0
                    entry: EmbeddingEntry
                    if single_pages is True:
                        for page in doc:
                            page_no += 1
                            descriptor_path = "{" + library_name + "}" +f"{rel_path}/{file}/page_{page_no}"
                            page_text = page.get_text()  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue, reportUnknownVariableType]
                            if isinstance(page_text, str) is False:
                                self.log.error(f"Can't read {rel_path} page {page_no}, ignoring page")
                                continue
                            page_text = cast(str, page_text)
                            if len(page_text) == 0:
                                continue
                            entry = {
                                'filename': file,
                                'text': page_text,
                                'page_no': page_no,
                                'emb_ten_idx': -1,
                                'emb_ten_size': -1
                            }
                            if descriptor_path in self.texts:
                                if self.texts[descriptor_path]['text'] == page_text:
                                    aborted = True  # Don't test for more pages, if page checked is identical
                                    break
                                else:
                                    self.log.info(f"Document page {descriptor_path} has been modified, recalculating")
                            self.texts[descriptor_path] = entry
                            count += 1
                        if aborted is False:
                            self.log.info(f"Read {rel_path}/{file}, {page_no - 1} pages.")
                    else:
                        pdf_text: str = ""
                        descriptor_path = "{" + library_name + "}" +f"{rel_path}/{file}"
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
                ext = os.path.splitext(file)
                if len(ext)==2:
                    ext = ext[1]
                else:
                    self.log.error(f"Can't identify extension of {file}, ignoring")
                    continue
                if ext in extensions:
                    rel_path = root[len(l_path):]
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
        emb_file = os.path.join(self.embeddings_path, f"library_embeddings.npz")
        desc_file = os.path.join(self.embeddings_path, f"library_desc.json")
        np.savez(emb_file, array=self.emb_ten)
        with open(desc_file, 'w') as f:
            json.dump(self.texts, f)
        self.log.info(f"Info saved {emb_file} and {desc_file}")

    def load_text_embeddings(self) -> int:
        emb_file = os.path.join(self.embeddings_path, f"library_embeddings.npz")
        desc_file = os.path.join(self.embeddings_path, f"library_desc.json")
        if os.path.exists(emb_file):
            self.emb_ten = np.load(emb_file)['array']
            with open(desc_file, 'r') as f:
                self.texts  = json.load(f)
        count = len(self.texts)
        if self.emb_ten is not None:
            self.log.info(f"Embeddings loaded: texts: {len(self.texts)}, emb_ten: {self.emb_ten.shape}")
        else:
            self.log.info(f"Embeddings loaded: texts: {len(self.texts)}")
        return count
                
    def gen_embeddings(self, model: str, library_name: str, verbose: bool=False, av_chunk_size: int=2048, av_chunk_overlay: int=0, chunk_method:str = "cut",  save_every_sec: float | None=360):
        lib_desc = '{' + library_name + '}'
        last_save = time.time()
        cnt: int = 0
        max_cnt = len(self.texts.keys())
        index = 0
        for desc in self.texts:
            if self.texts[desc]['emb_ten_idx'] != -1 and self.texts[desc]['emb_ten_size'] != -1:
                index = self.texts[desc]['emb_ten_idx'] + self.texts[desc]['emb_ten_size']
                cnt += 1
                # self.log.info(f"Skipping {desc}, already processed, {cnt}/{max_cnt}, index={index}")
                continue
            if desc.startswith(lib_desc):
                text: str = self.texts[desc]['text']
                if len(text) == 0:
                    self.log.warning(f"Text for {desc} is empty, ignoring!")
                    continue
                # text_chunks = [self.get_chunk(text, i) for i in range((len(text)-1) // chunk_size + 1) ]
                text_chunks = self.get_chunks(text, av_chunk_size, av_chunk_overlay, chunk_method)
                self.texts[desc]['emb_ten_idx'] = index
                self.texts[desc]['emb_ten_size'] = len(text_chunks)
                response = ollama.embed(model=model, input=text_chunks)
                embedding = np.asarray(response["embeddings"], dtype=np.float32)
                if len(embedding.shape)<2 or embedding.shape[0] != len(text_chunks):
                    self.log.error(f"Assumption on numpy conversion failed, can't generate embeddings for {desc}, result: {embedding.shape}, chunks: {len(text_chunks)}")
                    continue
                if self.emb_ten is None:
                    self.emb_ten = embedding
                else:
                    self.emb_ten = np.append(self.emb_ten, embedding, axis=0)  
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

    def yellow_line_it(self, model: str, text: str, search_embedding: numpy.typing.ArrayLike, context:int=16) -> list[float]:
        clr: list[str] = []
        for i in range(len(text)):
            i0 = i - context // 2
            i1 = i + context // 2
            if i0 < 0:
                i1 = i1 - i0
                i0 = 0
            if i1 > len(text):
                i0 = i0 - (i1 - len(text))
                i1 = len(text)
            clr.append(text[i0:i1])
        response = ollama.embed(model=model, input=clr)
        embs = np.asarray(response['embeddings'], dtype=np.float32)
        yellow: list[float] = []
        for i in range(embs.shape[0]):
            emb = embs[i,:]
            val: float = np.dot(search_embedding, emb) / np.linalg.norm(emb)
            yellow.append(val)
        return yellow

    def search_embeddings(self, model: str, search_text: str, verbose: bool=False, av_chunk_size: int=2048, av_chunk_overlay: int=0, chunk_method:str="cut", yellow_liner: bool=False, context:int=16, max_results:int=10) -> list[SearchResult] | None:
        response = ollama.embed(model=model, input=search_text)
        search_embedding = np.array(response["embeddings"][0], dtype=np.float32)
        search_embedding = search_embedding / np.linalg.norm(search_embedding)
        if len(search_text) > av_chunk_size:
            self.log.warning(f"Search text is longer than av_chunk_size: {len(search_text)}")
        if verbose is True:
            self.log.info(f"Search-embedding: {search_embedding.shape}")
        results: list[SearchResult] = []
        
        if self.emb_ten is None or self.texts == {}:
            return None
        t0 = time.time()
        idx_vec: np.typing.ArrayLike = np.asarray(np.matmul(self.emb_ten, search_embedding), dtype=np.float32)
        # arg_max = np.argmax(idx_vec)
        idx_list = cast(list[float], idx_vec.tolist())
        idx_idx = list(enumerate(idx_list))
        idx_srt = sorted(idx_idx, key=lambda x: x[1], reverse=True)

        # if idx_srt[0][0] != arg_max:
        #     print(f"That didn't work! {max_results}")
        # else:
        #     print("Wonderful!")
        
        for desc in self.texts:
            entry = self.texts[desc]
            for res_i in range(max_results):
                # index_i = idx_srt[res_i][0]
                arg_max_i = idx_srt[res_i][0]
                if entry['emb_ten_idx'] <= arg_max_i and entry['emb_ten_idx']+entry['emb_ten_size'] > arg_max_i:
                    self.log.info(f"{res_i}. {idx_vec[arg_max_i]}: {desc}")
                    index = arg_max_i - entry['emb_ten_idx']
                    chunk = self.get_chunk(entry['text'], index, av_chunk_size, av_chunk_overlay, chunk_method)
                    if yellow_liner is True:
                        yellow_liner_weights = self.yellow_line_it(model, chunk, search_embedding, context=context)
                    else:
                        yellow_liner_weights = None
                    result: SearchResult = {
                        'cosine': idx_vec[arg_max_i],
                        'desc': desc,
                        'index': index,
                        'text': entry['text'],
                        'chunk': chunk.replace('\n',' '),
                        'yellow_liner': yellow_liner_weights
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
