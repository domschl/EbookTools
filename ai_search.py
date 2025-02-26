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

class RepoState(TypedDict):
    embeddings_model_name: str
    chunk_size: int
    chunk_overlap: int
    text_entries: dict[str, EmbeddingsEntry]

# "nomic-ai/nomic-embed-text-v2-moe"
class HuggingfaceEmbeddings():
    def __init__(self, embeddings_model_name:str, repository:str, chunk_size:int, chunk_overlap:int) -> None:
        self.log: logging.Logger = logging.getLogger("HuggingfaceEmbedder")
        self.embeddings_matrix: torch.Tensor | None = None
        self.model_name: str = embeddings_model_name
        self.chunk_size: int = chunk_size
        if chunk_overlap >= chunk_size:
            self.log.error(f"chunk_overap={chunk_overlap} must be smaller than chunk_size={chunk_size}, using default 1/3 of chunk_size")
            self.chunk_overlap = chunk_size // 3
        else:
            self.chunk_overlap = chunk_overlap
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
            if torch.cuda.is_available():
                self.engine = self.engine.to(torch.device('cuda'))
            elif torch.backends.mps.is_available():
                self.engine = self.engine.to(torch.device('mps'))
            else:
                self.engine = self.engine.to(torch.device('cpu'))
            self.model_available = True
        except Exception as e:
            self.log.error(f"Huggingface engine {embeddings_model_name} not available: {e}")
            self.model_available = False
            self.engine = None

    def load_state(self) -> bool:
        ret: bool = True
        if self.repository_path is None:
            self.log.error("Cannot load state, since repository_path does not exist!")
            return False
        model_san = self.model_name.replace('/', '-')
        state_file = os.path.join(self.repository_path, f"texts_library_{model_san}_{self.chunk_size}|{self.chunk_overlap}.json")
        if os.path.exists(state_file) is False:
            self.log.error(f"Can't open {state_file}")
            ret = False
        else:
            repo_state: RepoState
            try:
                with open(state_file, 'r') as f:
                    repo_state = json.load(f)
            except Exception as e:
                self.log.error(f"Failed to load embeddings repository state, can't continue: {e}")
                exit(1)
            if repo_state['chunk_overlap'] != self.chunk_overlap or repo_state['chunk_size'] != self.chunk_size or self.model_name != repo_state['embeddings_model_name']:
                self.log.error("State-file is incompatible with model_name, or chunk_size, or chunk_overlap provided with initialization, can't continue")
                exit(1)
            self.texts = repo_state['text_entries']

        os.makedirs(self.pdf_cache_path, exist_ok=True)
        pdf_cache_index = os.path.join(self.pdf_cache_path, "pdf_index.json")
        if os.path.exists(pdf_cache_index):
            with open(pdf_cache_index, 'r') as f:
                self.pdf_index = json.load(f)
        else:
            self.pdf_index = {}
        os.makedirs(self.embeddings_path, exist_ok=True)
        embeddings_tensor_file = os.path.join(self.embeddings_path, f"embeddings_{model_san}.pt")
        if torch.cuda.is_available():
            map_location = torch.device('cuda')
        elif torch.backends.mps.is_available():
            map_location = torch.device('mps')
        else:
            map_location = torch.device('cpu')
        if os.path.exists(embeddings_tensor_file):
            self.embeddings_matrix = torch.load(embeddings_tensor_file, map_location=map_location)  # type: ignore
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
        return ret

    def save_pdf_cache_state(self):
        pdf_cache_index = os.path.join(self.pdf_cache_path, "pdf_index.json")
        with open(pdf_cache_index, 'w') as f:
            json.dump(self.pdf_index, f)

    def save_state(self) -> bool:
        if self.repository_path is None:
            self.log.error("Cannot save state, since repository_path does not exist!")
            return False
        model_san = self.model_name.replace('/', '-')
        state_file = os.path.join(self.repository_path, f"texts_library_{model_san}_{self.chunk_size}|{self.chunk_overlap}.json")
        repo_state = {
            'embeddings_model_name': self.model_name,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'text_entries': self.texts
        }
        with open(state_file, 'w') as f:
            json.dump(repo_state, f)
        if os.path.isdir(self.pdf_cache_path) is False:
            os.makedirs(self.pdf_cache_path, exist_ok=True)
        self.save_pdf_cache_state()
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

    def get_chunk_ptr(self, text: str, index: int) -> int:
        chunk_ptr = index*(self.chunk_size-self.chunk_overlap)
        return chunk_ptr

    def get_chunk(self, text: str, index: int) -> str:
        chunk_start = self.get_chunk_ptr(text, index)
        chunk = text[chunk_start : chunk_start + self.chunk_size]
        return chunk

    def get_span_chunk(self, text:str, index: int, count:int):
        if count < 1:
            self.log.error(f"Invalid multi_chunk count: {count}")
            return ""
        chunk_start = self.get_chunk_ptr(text, index)
        offset = self.chunk_size - self.chunk_overlap
        chunk = text[chunk_start : chunk_start + self.chunk_size+offset * (count-1)]
        return chunk

    def get_chunks(self, text:str) -> list[str]:
        chunks = (len(text) - 1) // (self.chunk_size - self.chunk_overlap) + 1 
        text_chunks = [self.get_chunk(text, i) for i in range(chunks) ]
        return text_chunks
            
    def add_texts(self, source_folder:str, library_name:str, formats:list[str] = ["pdf", "txt", "md"], use_pdf_cache:bool=True):
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
        cur_idx:int = 0
        if self.embeddings_matrix is not None:
            cur_idx = self.embeddings_matrix.shape[0]
        for desc in self.texts:
            if desc.startswith(lib_prefix):
                update_debris.append(desc)
        for desc in self.new_texts:
            entry_idx = self.new_texts[desc]['emb_ten_idx'] + self.new_texts[desc]['emb_ten_size']
            if entry_idx > cur_idx:
                cur_idx = entry_idx
        if 'pdf' in formats:
            pdf_cache = os.path.join(self.repository_path, 'PDF_Cache')
            os.makedirs(pdf_cache, exist_ok=True)
        else:
            pdf_cache = None
        home_path = os.path.expanduser("~")

        count:int = 0
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
                                    try:
                                        with open(self.pdf_index[desc]['filename'], 'r') as f:
                                            text = f.read()
                                    except Exception as e:
                                        self.log.warning(f"Failed to read PDF cache file for {desc}: {e}")
                                        del self.pdf_index[desc]
                                        text = None
                                else:
                                    self.log.info(f"PDF file {full_path} has changed, re-importing")
                                    del self.pdf_index[desc]
                            else:
                                self.log.info(f"{desc} is not in PDF cache (size: {len(self.pdf_index.keys())})")
                        if text is None:
                            self.log.info(f"Importing and caching PDF {full_path}")
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
                                if pdf_cache is not None:
                                    pdf_ind: PDFIndex = {
                                        'filename': os.path.join(pdf_cache, str(uuid.uuid4())),
                                        'file_size': os.path.getsize(full_path)
                                    }
                                    with open(pdf_ind['filename'], 'w') as f:
                                        f.write(text)
                                    self.pdf_index[desc] = pdf_ind
                                    self.save_pdf_cache_state()
                                    self.log.info(f"Added {desc} to PDF cache, size: {len(self.pdf_index.keys())}")
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
                    new_chunks = self.get_chunks(text)
                    self.chunks += new_chunks
                    self.new_texts[desc] = {
                        'filename': rel_path, 
                        'text': text,
                        'emb_ten_idx': cur_idx,
                        'emb_ten_size': len(new_chunks)
                    }
                    self.log.info(f"Adding {desc} at index {cur_idx}")
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
        idxs_new = sorted(idxs, key=lambda x: x[0])
        for i, idxi in enumerate(idxs):
            if idxs_new[i] != idxi:
                self.log.warning(f"At index {i}, reordering happened: {idxs_new[i]} {idxs[i]}")
        idxs = idxs_new
        for i, idx in enumerate(idxs):
            if idx[0] != last_idx:
                self.log.error(f"Algorithm failure at index {i} debris removal: {idx[0]} != {last_idx}")
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

    def search(self, search_text:str, max_results:int=2, yellow_liner:bool=False, context:int=16, context_steps:int=4, compress:str="none"):
        sorted_simil_all, search_embeddings = self.search_vect(search_text)
        sorted_simil = sorted_simil_all[:max_results]
        search_results: list[SearchResult] = []
        resolved_list: list[tuple[str, int, int, float, EmbeddingsEntry]] = []
        yellow_liner_weights: np.typing.NDArray[np.float32] | None
        for result in sorted_simil:
            idx = result[0]
            cosine = result[1]
            for desc in self.texts:
                entry = self.texts[desc]
                if idx >= entry['emb_ten_idx'] and idx < entry['emb_ten_idx'] + entry['emb_ten_size']:
                    print(f"{desc}: {cosine}")
                    resolved_list.append((desc, idx, 1, cosine, entry))
        srla = sorted(resolved_list)
        for ind, sra in reversed(list(enumerate(srla))):
            if ind+1 == len(srla):
                continue
            if sra[0] == srla[ind+1][0]:  # same desc
                if sra[4]['emb_ten_idx'] + sra[4]['emb_ten_size'] >= srla[ind+1][4]['emb_ten_idx']:  # Overlapping consequtive
                    del srla[ind+1]
                    cnt:int = srla[ind][2] + 1
                    cosine: float = sra[3]
                    if sra[3] < srla[ind+1][3]:
                        cosine = srla[ind+1][3]  # get the better score
                    srla[ind] = (srla[ind][0], srla[ind][1], cnt, cosine, srla[ind][4])
                    self.log.info(f"Merged two consequtive search postions into a span-chunk for {sra[0]}")
        for sra in srla:
            desc, idx, count, cosine, entry = sra
            chunk:str = self.get_span_chunk(entry['text'], idx - entry['emb_ten_idx'], count)
            if compress == "light":
                new_chunk = chunk
                old_chunk = None
                while new_chunk != old_chunk:
                    old_chunk = new_chunk
                    new_chunk = old_chunk.replace("  ", " ").replace("\n\n", "\n")
                chunk = new_chunk
            elif compress == "full":
                new_chunk = chunk
                old_chunk = None
                while new_chunk != chunk:
                    old_chunk = new_chunk
                    new_chunk = old_chunk.replace("  ", " ").replace("\n\n", " ")
                chunk = new_chunk
            if yellow_liner is True:
                yellow_liner_weights = self.yellow_line_it(chunk, search_embeddings, context=context, context_steps=context_steps)
            else:
                yellow_liner_weights = None
            sres:SearchResult = {
                'cosine': cosine,
                'index': idx,
                'offset': idx - entry['emb_ten_idx'],
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
