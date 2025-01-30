import logging
import os
import json
import time

import ollama
import numpy as np


class EmbeddingSearch:
    def __init__(self, embeddings_path, epsilon = 1e-6):
        self.log: logging.Logger = logging.getLogger("EmbSearch")
        modes = ["filepath", "textlibrary"]
        self.epsilon = epsilon
        self.np_type = np.float32
        self.texts = None
        self.repos = {}
        e_path = os.path.expanduser(embeddings_path)
        if os.path.exists(e_path) is False:
            self.log.error(f"embeddings_path={embeddings_path} does not exist!")
        self.embeddings_path = e_path
        self.load_text_embeddings()
        self.load_repos()

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
                        entry = {
                            'filename': file,
                            'text': doc_text
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
    def get_chunk(text, index, chunk_size=2048):
        chunk = text[index*chunk_size:(index+1)*chunk_size]
        return chunk

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)
        
    def save_text_embeddings(self):
        emb_file = os.path.join(self.embeddings_path, 'library_embeddings.json')
        with open(emb_file, 'w') as f:
            json.dump(self.texts, f, cls=self.NumpyEncoder)

    def load_text_embeddings(self, silent=True):
        emb_file = os.path.join(self.embeddings_path, 'library_embeddings.json')
        if os.path.exists(emb_file) is False:
            if silent is False:
                self.log.error(f"Embeddings file {emb_file} does not exist!")
            return 0
        with open(emb_file, 'r') as f:
            self.texts  = json.load(f)
        for txt in self.texts:
            for key in self.texts[txt]:
                if key.startswith('emb_'):
                    self.texts[txt][key] = np.asarray(self.texts[txt][key], dtype=self.np_type)
        return len(self.texts)
                
    def gen_embeddings(self, model, library_name=None, verbose=False, chunk_size=2048, save_every_sec=60):
        if library_name is not None:
            lib_desc = '{' + library_name + '}'
        else:
            lib_desc = None
        last_save = time.time()
        emb_key = f"emb_{model}"
        if self.texts is None:
            return
        for desc in self.texts:
            if library_name is None or desc.startswith(lib_desc):
                if emb_key not in self.texts[desc]:
                    text = self.texts[desc]['text']
                    text_chunks = [self.get_chunk(text, i) for i in range(len(text) // chunk_size)]
                    embeddings = None
                    for text_chunk in text_chunks:
                        response = ollama.embed(model=model, input=text_chunk)
                        embedding = np.array(response["embeddings"][0], dtype=self.np_type)
                        if embeddings is None:
                            embeddings = np.array([embedding], dtype=self.np_type)
                        else:
                            embeddings = np.append(embeddings, np.array([embedding], dtype=self.np_type), axis=0)
                        print(".", end="")
                    print()   
                    self.texts[desc][emb_key] = embeddings
                    # del self.texts[desc]['text']
                    if verbose is True:
                         print(f"Generated {self.texts[desc][emb_key].shape[0]} embeddings for {desc}")
                    if save_every_sec is not None:
                        if time.time() - last_save > save_every_sec:
                            self.save_text_embeddings()
                            last_save = time.time()
        self.save_text_embeddings()

    def cos_sim(self, a, b):
        m = np.dot(a,b)
        n = np.linalg.norm(a) * np.linalg.norm(b)
        if n < self.epsilon:
            n = self.epsilon
        return m / n

    def search_embeddings(self, model, search_text, verbose=False, chunk_size=2048):
        search_text = search_text[:chunk_size]
        while len(search_text) < chunk_size:
            search_text = " " + search_text
        response = ollama.embed(model=model, input=search_text)
        search_embedding = np.array(response["embeddings"][0], dtype=self.np_type)
        best_doc = ""
        best_index = -1
        cos_val = 0
        best_chunk = ""
        if self.texts is None:
            return
        for desc in self.texts:
            entry =  self.texts[desc]
            emb_key = f"emb_{model}"
            if emb_key not in entry:
                if verbose is True:
                    print(f"No embeddings from model {model} in {desc['filename']}, ignoring doc")
                continue
            for index in range(entry[emb_key].shape[0]):
                chunk = entry[emb_key][index,:]
                # print("Search:", list(search_embedding))
                # print("Chunk:", list(chunk))
                # if index == 1:
                #     return
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
