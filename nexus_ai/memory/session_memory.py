from typing import List, Any
from autogen_core.memory import Memory, MemoryContent
from datetime import datetime

class SessionMemory(Memory):
    
    def __init__(self, max_turns: int = 50):
        
        self._memory: List[MemoryContent] = []
        self._max_turns = max_turns
        
    async def add(self, content: MemoryContent) -> None:
        if content.metadata is None:
            content.metadata = {}
        content.metadata["timestamp"] = datetime.now().isoformat()
        
        self._memory.append(content)
        
        if len(self._memory) > self._max_turns:
            self._memory = self._memory[-self._max_turns:]
    
    async def query(self, query: str) -> List[MemoryContent]:
        return self._memory.copy()
    
    async def clear(self) -> None:
        self._memory.clear()
    
    async def close(self) -> None:
        pass
    
    async def update_context(self, model_context: Any) -> None:
        pass
    
    def get_recent(self, n: int = 5) -> List[MemoryContent]:
        return self._memory[-n:] if n < len(self._memory) else self._memory.copy()
    
    def __len__(self) -> int:
        return len(self._memory)