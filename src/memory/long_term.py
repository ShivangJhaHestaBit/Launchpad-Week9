import sqlite3
import json
from typing import List, Optional, Dict, Any
from autogen_core.memory import Memory, MemoryContent, MemoryMimeType

class LongTermMemory(Memory):
    def __init__(self, db_path: str = "long_term.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                mime_type TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                importance INTEGER DEFAULT 0
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_memory_type 
            ON memories(memory_type)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_importance 
            ON memories(importance DESC)
        """)
        
        conn.commit()
        conn.close()
    
    async def add(
        self,
        content: MemoryContent,
        memory_type: str = "episodic",
        importance: int = 0
    ) -> None:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        metadata_json = json.dumps(content.metadata) if content.metadata else None
        
        cursor.execute("""
            INSERT INTO memories (content, memory_type, mime_type, metadata, importance)
            VALUES (?, ?, ?, ?, ?)
        """, (
            content.content,
            memory_type,
            content.mime_type.value if isinstance(content.mime_type, MemoryMimeType) else content.mime_type,
            metadata_json,
            importance
        ))
        
        conn.commit()
        conn.close()
    
    async def query(
        self,
        query: str,
        memory_type: Optional[str] = None,
        limit: int = 10
    ) -> List[MemoryContent]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        sql = """
            SELECT content, mime_type, metadata
            FROM memories
            WHERE content LIKE ?
        """
        params = [f"%{query}%"]
        
        if memory_type:
            sql += " AND memory_type = ?"
            params.append(memory_type)
        
        sql += " ORDER BY importance DESC, created_at DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        conn.close()
        
        results = []
        for content, mime_type, metadata_json in rows:
            metadata = json.loads(metadata_json) if metadata_json else None
            results.append(MemoryContent(
                content=content,
                mime_type=mime_type,
                metadata=metadata
            ))
        
        return results
    
    async def get_important_memories(
        self,
        min_importance: int = 5,
        limit: int = 20
    ) -> List[MemoryContent]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT content, mime_type, metadata
            FROM memories
            WHERE importance >= ?
            ORDER BY importance DESC, created_at DESC
            LIMIT ?
        """, (min_importance, limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        results = []
        for content, mime_type, metadata_json in rows:
            metadata = json.loads(metadata_json) if metadata_json else None
            results.append(MemoryContent(
                content=content,
                mime_type=mime_type,
                metadata=metadata
            ))
        
        return results
    
    async def clear(self) -> None:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM memories")
        conn.commit()
        conn.close()
    
    async def close(self) -> None:
        pass
    
    async def update_context(self, model_context: Any) -> None:
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM memories")
        total = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM memories WHERE memory_type = 'episodic'")
        episodic = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM memories WHERE memory_type = 'semantic'")
        semantic = cursor.fetchone()[0]
        
        cursor.execute("SELECT AVG(importance) FROM memories")
        avg_importance = cursor.fetchone()[0] or 0
        
        conn.close()
        
        return {
            "total_memories": total,
            "episodic": episodic,
            "semantic": semantic,
            "avg_importance": round(avg_importance, 2)
        }