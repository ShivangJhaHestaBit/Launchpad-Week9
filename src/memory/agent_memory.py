from typing import List, Optional, Any
from autogen_core.memory import Memory, MemoryContent, MemoryMimeType
from autogen_core.models import UserMessage
from .session_memory import SessionMemory
from .vector_store import FAISSVectorMemory
from .long_term import LongTermMemory

class AgentMemorySystem(Memory):
    def __init__(
        self,
        session_max_turns: int = 50,
        vector_k: int = 5,
        vector_threshold: float = 0.3,
        db_path: str = "long_term.db",
        vector_persist_path: Optional[str] = "vector_store.faiss"
    ):
        self.session = SessionMemory(max_turns=session_max_turns)
        self.vector = FAISSVectorMemory(
            k=vector_k,
            score_threshold=vector_threshold,
            persist_path=vector_persist_path
        )
        self.long_term = LongTermMemory(db_path=db_path)
    
    async def add(self, content: MemoryContent, store_long_term: bool = False) -> None:

        await self.session.add(content)
        await self.vector.add(content)

        if store_long_term:
            importance = content.metadata.get("importance", 0) if content.metadata else 0
            memory_type = content.metadata.get("type", "episodic") if content.metadata else "episodic"
            await self.long_term.add(content, memory_type=memory_type, importance=importance)
    
    async def query(self, query: str) -> List[MemoryContent]:
        
        vector_results = await self.vector.query(query)
        
        lt_results = await self.long_term.query(query, limit=5)
        
        combined = vector_results + lt_results
        
        seen = set()
        unique_results = []
        for memory in combined:
            if memory.content not in seen:
                seen.add(memory.content)
                unique_results.append(memory)
        
        return unique_results
    
    async def get_context_for_query(self, query: str) -> List[MemoryContent]:
        context = []
        
        recent = self.session.get_recent(n=5)
        context.extend(recent)
        
        similar = await self.vector.query(query)
        context.extend(similar)
        
        important = await self.long_term.get_important_memories(min_importance=7, limit=5)
        context.extend(important)
        
        return context
    
    async def save_important_fact(
        self,
        fact: str,
        importance: int = 8,
        metadata: Optional[dict] = None
    ) -> None:
        if metadata is None:
            metadata = {}
        metadata["importance"] = importance
        metadata["type"] = "semantic"
        
        content = MemoryContent(
            content=fact,
            mime_type=MemoryMimeType.TEXT,
            metadata=metadata
        )
        await self.vector.add(content)
        await self.long_term.add(content, memory_type="semantic", importance=importance)
    
    async def clear_session(self) -> None:
        await self.session.clear()
    
    async def clear(self) -> None:
        await self.session.clear()
        await self.vector.clear()
        await self.long_term.clear()
    
    async def close(self) -> None:
        await self.session.close()
        await self.vector.close()
        await self.long_term.close()
    
    async def update_context(self, model_context: Any) -> None:
        memory_parts = []
        session_memories = await self.session.query("")
        if session_memories:
            recent = session_memories[-5:]
            if recent:
                session_text = "Recent conversation:\n"
                for i, mem in enumerate(recent, 1):
                    session_text += f"{i}. {mem.content}\n"
                memory_parts.append(session_text)
        
        important = await self.long_term.get_important_memories(min_importance=7, limit=5)
        if important:
            facts_text = "\nImportant facts:\n"
            for i, mem in enumerate(important, 1):
                facts_text += f"{i}. {mem.content}\n"
            memory_parts.append(facts_text)
        
        if memory_parts:
            combined_text = "\nRelevant memory content:\n" + "\n".join(memory_parts)
            await model_context.add_message(UserMessage(content=combined_text,source="memory"))
    
    def get_memory_stats(self) -> dict:
        """Get statistics from all memory stores"""
        return {
            "session": {
                "size": len(self.session)
            },
            "vector": {
                "size": len(self.vector)
            },
            "long_term": self.long_term.get_stats()
        }