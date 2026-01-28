from typing import List, Optional, Any
import faiss
from autogen_core.memory import Memory, MemoryContent
from sentence_transformers import SentenceTransformer
import pickle
import os


class FAISSVectorMemory(Memory):
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        k: int = 5,
        score_threshold: float = 0.3,
        persist_path: Optional[str] = None
    ):
        self._encoder = SentenceTransformer(embedding_model)
        self._k = k
        self._score_threshold = score_threshold
        self._persist_path = persist_path
        
        self._dimension = self._encoder.get_sentence_embedding_dimension()
        
        self._index = faiss.IndexFlatL2(self._dimension)
        
        self._contents: List[MemoryContent] = []
        
        if persist_path and os.path.exists(persist_path):
            self._load()
    
    async def add(self, content: MemoryContent) -> None:
        text = content.content
        embedding = self._encoder.encode([text], convert_to_numpy=True)
        
        faiss.normalize_L2(embedding)
        
        self._index.add(embedding)
        
        self._contents.append(content)
        
        if self._persist_path:
            self._save()
    
    async def query(self, query: str) -> List[MemoryContent]:
        if len(self._contents) == 0:
            return []
        
        query_embedding = self._encoder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        k = min(self._k, len(self._contents))
        distances, indices = self._index.search(query_embedding, k)
        
        similarities = 1 - (distances[0] ** 2) / 2
        
        results = []
        for idx, score in zip(indices[0], similarities):
            if score >= self._score_threshold:
                content = self._contents[idx]
                if content.metadata is None:
                    content.metadata = {}
                content.metadata["similarity_score"] = float(score)
                results.append(content)
        
        return results
    
    async def clear(self) -> None:
        self._index.reset()
        self._contents.clear()
        
        if self._persist_path and os.path.exists(self._persist_path):
            os.remove(self._persist_path)
            metadata_path = f"{self._persist_path}.meta"
            if os.path.exists(metadata_path):
                os.remove(metadata_path)
    
    async def close(self) -> None:
        if self._persist_path:
            self._save()
    
    async def update_context(self, model_context: Any) -> None:
        pass
    
    def _save(self) -> None:
        if not self._persist_path:
            return
        
        faiss.write_index(self._index, self._persist_path)
        
        metadata_path = f"{self._persist_path}.meta"
        with open(metadata_path, 'wb') as f:
            pickle.dump(self._contents, f)
    
    def _load(self) -> None:
        if not self._persist_path or not os.path.exists(self._persist_path):
            return
        
        self._index = faiss.read_index(self._persist_path)
        
        metadata_path = f"{self._persist_path}.meta"
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                self._contents = pickle.load(f)
    
    def __len__(self) -> int:
        return len(self._contents)