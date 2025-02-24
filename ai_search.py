import logging
import os
import json
from typing import TypedDict, cast
import numpy as np
import pymupdf  # pyright: ignore[reportMissingTypeStubs]
import torch
from sentence_transformers import SentenceTransformer
import uuid


class EmbeddingsEntry(TypedDict):
    filename: str
    text: str
    emb_ten_idx: int
    emb_ten_size: int

class SearchResult(TypedDict):
    cosine: float
    index: int
    offset: int
    desc: str
    text: str
    chunk: str
    yellow_liner: np.typing.NDArray[np.float32] | None

class PDFIndex(TypedDict):
    filename: str
    file_size: int


# "nomic-ai/nomic-embed-text-v2-moe"
class HuggingfaceEmbeddings():
    def __init__(self, embeddings_model_name:str, repository:str):
        self.log: logging.Logger = logging.getLogger("HuggingfaceEmbedder")
        self.embeddings_matrix: torch.Tensor | None = None
        self.model_name: str = embeddings_model_name
        if os.path.isdir(repository) is False:
            self.log.error(f"Repository {repository} does not exist!")
            self.repository_path = None
        else:
            self.repository_path = repository
            self.pdf_cache_path = os.path.join(self.repository_path, "PDF_Cache")
            self.embeddings_path = os.path.join(self.repository_path, "embeddings")

        self.texts:dict[str, EmbeddingsEntry] = {}
        self.new_texts:dict[str, EmbeddingsEntry] = {}
        self.pdf_index:dict[str, PDFIndex] = {}
        self.chunks:list[str] = []
        self.debris:list[str] = []
        try:
            self.engine: SentenceTransformer | None = SentenceTransformer(embeddings_model_name, trust_remote_code=True)
            self.model_available = True
        except Exception as e:
            self.log.error(f"Huggingface engine {embeddings_model_name} not available: {e}")
            self.model_available = False
            self.engine = None

    def load_state(self) -> bool:
        if self.repository_path is None:
            self.log.error("Cannot load state, since repository_path does not exist!")
            return False
        model_san = self.model_name.replace('/', '-')
        state_file = os.path.join(self.repository_path, f"texts_library_{model_san}.json")
        if os.path.exists(state_file) is False:
            self.log.error(f"Can't open {state_file}")
            return False
        with open(state_file, 'r') as f:
            self.texts = json.load(f)
        os.makedirs(self.pdf_cache_path, exist_ok=True)
        pdf_cache_index = os.path.join(self.pdf_cache_path, "pdf_index.json")
        if os.path.exists(pdf_cache_index):
            with open(pdf_cache_index, 'r') as f:
                self.pdf_index = json.load(f)
        else:
            self.pdf_index = {}
        os.makedirs(self.embeddings_path, exist_ok=True)
        embeddings_tensor_file = os.path.join(self.embeddings_path, f"embeddings_{model_san}.pt")
        if os.path.exists(embeddings_tensor_file):
            self.embeddings_matrix = torch.load(embeddings_tensor_file)  # type: ignore
        else:
            self.embeddings_matrix = None
        if self.embeddings_matrix is not None:
            sum = 0
            for desc in self.texts:
                sum += self.texts[desc]['emb_ten_size']
            self.log.info(f"Matrix: {self.embeddings_matrix.shape}, chunks: {sum}, texts: {len(self.texts.keys())}")
            if sum != self.embeddings_matrix.shape[0]:
                self.log.error("Embeddings-matrix and text chunks have diverged!")
                exit(1)
        else:
            self.log.warning("No embeddings available!")
        return True

    def save_state(self) -> bool:
        if self.repository_path is None:
            self.log.error("Cannot save state, since repository_path does not exist!")
            return False
        model_san = self.model_name.replace('/', '-')
        state_file = os.path.join(self.repository_path, f"texts_library_{model_san}.json")
        with open(state_file, 'w') as f:
            json.dump(self.texts, f)
        if os.path.isdir(self.pdf_cache_path) is False:
            os.makedirs(self.pdf_cache_path, exist_ok=True)
        pdf_cache_index = os.path.join(self.pdf_cache_path, "pdf_index.json")
        with open(pdf_cache_index, 'w') as f:
            json.dump(self.pdf_index, f)
        if os.path.isdir(self.embeddings_path) is False:
            os.makedirs(self.embeddings_path, exist_ok=True)
        embeddings_tensor_file = os.path.join(self.embeddings_path, f"embeddings_{model_san}.pt")
        if self.embeddings_matrix is not None:
            torch.save(self.embeddings_matrix, embeddings_tensor_file)  # type: ignore
        return True

    def embed(self, text_chunks: list[str], description:str | None=None, append:bool=False):
        if self.model_available is False:
            self.log.error("Embeddings model is not available")
            return torch.Tensor([])
        if description is not None:
            self.log.info(f"Generating embedding for {description}...")
        if self.engine is None:
            self.log.error("Embeddings engine is not available")
            return torch.Tensor([])
        self.log.info(f"Encoding {len(text_chunks)} chunks...")
        if len(text_chunks) == 0:
            self.log.error("Cannot encode empty text list!")
            return torch.tensor([])
        embeddings: list[torch.Tensor] = self.engine.encode(sentences=text_chunks, show_progress_bar=True, convert_to_numpy=False)  # type: ignore  # API is a mess!
        emb_matrix = torch.stack(embeddings)  # pyright: ignore[reportUnknownArgumentType]
        if append is True:
            if self.embeddings_matrix is None:
                self.embeddings_matrix = emb_matrix
            else:
                self.embeddings_matrix = torch.cat([self.embeddings_matrix, emb_matrix])
        return emb_matrix

    def search_vect(self, text:str) -> tuple[list[tuple[int, float]], torch.Tensor]:
        if self.embeddings_matrix is None:
            self.log.error("No embeddings available!")
            return [], torch.Tensor([])
        vect: list[torch.Tensor] = self.engine.encode([text], show_progress_bar=True, convert_to_numpy=False)  # type:ignore
        if len(vect) == 0:
            self.log.error("Failed to calculate embedding for search")
            return [], torch.Tensor([])
        if len(vect) > 1:
            self.log.warning("Result contains more than one vector, ignoring additional ones")
        search_vect:torch.Tensor = vect[0]
        simil = enumerate(torch.matmul(self.embeddings_matrix, search_vect).cpu().numpy().tolist())  # type:ignore
        sorted_simil:list[tuple[int, float]] = sorted(simil, key=lambda x: x[1], reverse=True)  # type:ignore
        return sorted_simil, search_vect

    def get_chunk(self, text: str, index: int, chunk_size: int=2048, chunk_overlap:int=0) -> str:
        chunk = text[index*(chunk_size-chunk_overlap):(index+1)*(chunk_size-chunk_overlap)]
        return chunk

    def get_chunks(self, text:str, chunk_size:int=2048, chunk_overlap:int=0) -> list[str]:
        chunks = (len(text) - 1) // (chunk_size - chunk_overlap) + 1 
        text_chunks = [self.get_chunk(text, i) for i in range(chunks) ]
        return text_chunks
            
    def add_texts(self, source_folder:str, library_name:str, formats:list[str] = ["pdf", "txt", "md"], use_pdf_cache:bool=True, chunk_size:int=2048, chunk_overlap:int=1024):
        if self.repository_path is None:
            self.log.error("Cannot add texts, since repository path does not exist")
            return 0
        source_path = os.path.abspath(os.path.expanduser(source_folder))
        lib_prefix = "{" + library_name + "}"
        update_debris: list[str] = []
        known_formats = ["pdf", "txt", "md", "py"]
        for format in formats:
            if format not in known_formats:
                self.log.error(f"Format {format} is not supported, removing")
                formats.remove(format)
        for desc in self.texts:
            if desc.startswith(lib_prefix):
                update_debris.append(desc)
        if 'pdf' in formats:
            pdf_cache = os.path.join(self.repository_path, 'PDF_Cache')
            os.makedirs(pdf_cache, exist_ok=True)
        home_path = os.path.expanduser("~")
        cur_idx:int = 0
        count:int = 0
        if self.embeddings_matrix is not None:
            cur_idx = self.embeddings_matrix.shape[0]
        self.log.info(f"Adding texts from {source_folder} of formats {formats}")
        for root, _dir, files in os.walk(source_path):
            for file in files:
                ext = os.path.splitext(file)
                if len(ext)==2:
                    ext = ext[1]
                    if len(ext) > 0:
                        ext = ext[1:]
                else:
                    self.log.error(f"Can't identify extension of {file}, ignoring")
                    continue
                if ext in formats:
                    dupe: bool = False
                    full_path = os.path.join(root, file)
                    if full_path.startswith(home_path):
                        rel_path = "~" + full_path[len(home_path):]
                    else:
                        rel_path = full_path
                    desc = lib_prefix + full_path[len(source_path):]
                    self.log.info(f"Adding: {rel_path} as {desc}")
                    if desc in update_debris:
                        update_debris.remove(desc)
                        dupe = True
                    if ext == "pdf":
                        text: str | None = None
                        if use_pdf_cache is True:
                            if desc in self.pdf_index:
                                if os.path.getsize(full_path) == self.pdf_index[desc]['file_size']:
                                    with open(self.pdf_index[desc]['filename'], 'r') as f:
                                        text = f.read()
                        if text is None:
                            doc = pymupdf.open(full_path)
                            text = ""
                            for page in doc:
                                page_text = page.get_text()  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue, reportUnknownVariableType]
                                if isinstance(page_text, str) is False:
                                    self.log.error(f"Can't read page of {full_path}, ignoring page")
                                    continue
                                page_text = cast(str, page_text)
                                if len(page_text) == 0:
                                    continue
                                text += page_text
                            if text == "":
                                text = None
                            else:
                                pdf_ind: PDFIndex = {
                                    'filename': str(uuid.uuid4()),
                                    'file_size': os.path.getsize(full_path)
                                }
                                with open(pdf_ind['filename'], 'w') as f:
                                    f.write(text)
                                self.pdf_index[desc] = pdf_ind
                    else:  # Text format
                        with open(full_path, 'r') as f:
                            text = f.read()
                    if text is None:
                        continue
                    if dupe is True:
                        if self.texts[desc]['text'] == text:
                            continue
                        else:
                            update_debris.append(desc)  # re-add, changed text
                    new_chunks = self.get_chunks(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                    self.chunks += new_chunks
                    self.new_texts[desc] = {
                        'filename': rel_path, 
                        'text': text,
                        'emb_ten_idx': cur_idx,
                        'emb_ten_size': len(new_chunks)
                    }
                    count += 1
                    cur_idx += len(new_chunks)
                else:
                    pass
                    # self.log.info(f"Ignoring: {file}, {ext} not in {formats}")
        self.debris += update_debris
        return count
    
    def generate_embeddings(self, description:str=""):
        if len(self.debris) > 0:
            shift_new = 0
            for deb in self.debris:
                if deb not in self.texts:
                    self.log.error(f"Debris {deb} got lost, its not in texts!")
                    continue
                if deb in self.pdf_index:
                    os.remove(self.pdf_index[deb]['filename'])
                    del self.pdf_index[deb]
                emb:EmbeddingsEntry = self.texts[deb]
                start:int = emb['emb_ten_idx']
                length:int = emb['emb_ten_size']
                shift_new += length
                if self.embeddings_matrix is None:
                    self.log.error("Embeddings not existing, but there's debris!")
                    return
                self.embeddings_matrix = torch.cat([self.embeddings_matrix[:start,:], self.embeddings_matrix[start+length:,:]])
                for txt in self.texts:
                    if self.texts[txt]['emb_ten_idx'] > start:
                        self.texts[txt]['emb_ten_idx'] -= length
                del self.texts[deb]
                self.log.info(f"Debris {deb} cleaned up")
            for desc in self.new_texts:
                self.new_texts[desc]['emb_ten_idx'] -= shift_new
        self.texts.update(self.new_texts)
        self.new_texts = {}

        # Calc new embeddings
        self.debris = []            
        self.embed(self.chunks, description, append=True)
        self.chunks = []

        # Check data consistency
        idxs:list[tuple[int,int]] = []
        for desc in self.texts:
            entry = self.texts[desc]
            idxs.append((entry['emb_ten_idx'], entry['emb_ten_size']))
        last_idx = 0
        idxs = sorted(idxs, key=lambda x: x[0])
        for idx in idxs:
            if idx[0] != last_idx:
                self.log.error(f"Algorithm failure at debris removal: {idx[0]} != {last_idx}")
                exit(1)
            last_idx += idx[1]
        if self.embeddings_matrix is None:
            self.log.error("No embeddings available, but text debris found, algorithm failure")
            exit(1)
        if last_idx != self.embeddings_matrix.shape[0]:
            self.log.error(f"Length of embeddings-tensor (dim-0) {self.embeddings_matrix.shape} != {last_idx} (last-idx), algorithm failure")
            exit(1)

    def yellow_line_it(self, text: str, search_embeddings: torch.Tensor, context:int=16, context_steps:int=1) -> np.typing.NDArray[np.float32]:
        if self.embeddings_matrix is None:
            self.log.error("No embeddings available at yellow-lining!")
            return np.array([], dtype=np.float32)
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
        embs = self.embed(clr, append=False)
        yellow_vect: np.typing.NDArray[np.float32] = torch.matmul(embs, search_embeddings).cpu().numpy()  # type: ignore
        return yellow_vect

    def search(self, search_text:str, max_results:int=2, chunk_size:int=2048, chunk_overlap:int=1024, yellow_liner:bool=False, context:int=16, context_steps:int=4):
        sorted_simil_all, search_embeddings = self.search_vect(search_text)
        sorted_simil = sorted_simil_all[:max_results]
        search_results: list[SearchResult] = []
        yellow_liner_weights: np.typing.NDArray[np.float32] | None
        for result in sorted_simil:
            idx = result[0]
            cosine = result[1]
            for desc in self.texts:
                entry = self.texts[desc]
                if idx >= entry['emb_ten_idx'] and idx < entry['emb_ten_idx'] + entry['emb_ten_size']:
                    chunk = self.get_chunk(entry['text'], idx - entry['emb_ten_idx'], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                    if yellow_liner is True:
                        yellow_liner_weights = self.yellow_line_it(chunk, search_embeddings, context=context, context_steps=context_steps)
                    else:
                        yellow_liner_weights = None
                    sres:SearchResult = {
                        'cosine': cosine,
                        'index': idx,
                        'offset': entry['emb_ten_idx'] - idx,
                        'desc': desc,
                        'chunk': chunk,  
                        'text': entry['text'],
                        'yellow_liner': yellow_liner_weights
                    }
                    search_results.append(sres)
        return search_results
    

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
