import os
import hashlib
from typing import List, Dict, Optional
from datetime import datetime

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from .base import BaseMemoryManager


class CustomMemoryManager(BaseMemoryManager):
    """Memory manager using local ChromaDB"""
    
    def __init__(
        self,
        memory_db_path: str = "data/memory_store",
        short_term_size: int = 10,
        embedding_model: str = "mixedbread-ai/mxbai-embed-large-v1",
        profile_path: str = "data/memory_store/user_profile.json"
    ):
        super().__init__(short_term_size=short_term_size, profile_path=profile_path)
        
        self.memory_db_path = memory_db_path
        
        if not os.path.exists(memory_db_path):
            try:
                os.makedirs(memory_db_path, exist_ok=True)
                print(f"    Created memory directory at {memory_db_path}")
            except Exception as e:
                print(f"    Failed to create memory directory: {e}")

        print(f"   Initializing Custom Memory (ChromaDB)")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        print(f"   Memory embeddings: {embedding_model}")

        try:
            self.semantic_memory = Chroma(
                persist_directory=memory_db_path,
                embedding_function=self.embeddings
            )
            print(f"     Custom memory store initialized at {memory_db_path}")
        except Exception as e:
            print(f"     Failed to initialize ChromaDB: {e}")
            self.semantic_memory = None

    def store_conversation(
        self,
        user_message: str,
        agent_response: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """Store conversation in semantic memory"""
        if not self.semantic_memory:
             return "failed_init"
             
        timestamp = datetime.now().isoformat()
        
        conv_id = hashlib.md5(f"{user_message}{timestamp}".encode()).hexdigest()[:12]
        
        meta = {
            "conversation_id": conv_id,
            "timestamp": timestamp,
            "user_message": user_message,
            "agent_response": agent_response[:500] if agent_response else "",
        }
        
        if metadata:
            meta.update(metadata)
        
        combined_text = f"User Question: {user_message}\n\nAgent Response: {agent_response}"
        
        doc = Document(
            page_content=combined_text,
            metadata=meta
        )
        
        try:
            self.semantic_memory.add_documents([doc])
            print(f"     Stored conversation in semantic memory (ID: {conv_id})")
            return conv_id
        except Exception as e:
             print(f"     Error storing conversation: {e}")
             return "error"

    def retrieve_relevant_memories(
        self,
        query: str,
        k: int = 5,
        recency_weight: float = 0.3,
        relevance_weight: float = 0.5,
        importance_weight: float = 0.2
    ) -> List[Dict]:
        """Retrieve memories with hybrid scoring"""
        if not self.semantic_memory:
            return []
            
        try:
            results = self.semantic_memory.similarity_search_with_relevance_scores(
                query,
                k=k * 2
            )
            
            if not results:
                return []
            
            scored_memories = []
            now = datetime.now()
            
            for doc, relevance_score in results:
                timestamp_str = doc.metadata.get("timestamp", "")
                try:
                    timestamp = datetime.fromisoformat(timestamp_str)
                    days_old = (now - timestamp).days
                    recency_score = max(0, 1 - (days_old / 30))
                except:
                    recency_score = 0.0
                
                quality = doc.metadata.get("citation_quality", "Good")
                importance_score = {
                    "Excellent": 1.0, "Good": 0.8, "Fair": 0.6, "Poor": 0.4
                }.get(quality, 0.5)
                
                hybrid_score = (
                    relevance_score * relevance_weight +
                    recency_score * recency_weight +
                    importance_score * importance_weight
                )
                
                scored_memories.append({
                    "user_message": doc.metadata.get("user_message", ""),
                    "agent_response": doc.metadata.get("agent_response", ""),
                    "memory_text": f"User: {doc.metadata.get('user_message', '')} | Agent: {doc.metadata.get('agent_response', '')[:100]}",
                    "timestamp": timestamp_str,
                    "conversation_id": doc.metadata.get("conversation_id", ""),
                    "relevance_score": relevance_score,
                    "hybrid_score": hybrid_score,
                    "metadata": doc.metadata
                })
            
            # sort by hybrid score
            scored_memories.sort(key=lambda x: x["hybrid_score"], reverse=True)
            
            return scored_memories[:k]
            
        except Exception as e:
            print(f"     Error retrieving memories: {e}")
            return []

    def get_all_memories(self) -> List[Dict]:
        """Get all memories (IDs only for count, or full docs if needed)"""
        if not self.semantic_memory:
            return []
        try:
             data = self.semantic_memory.get()
             if not data:
                 return []
             
             ids = data.get("ids", [])
             return [{"id": i} for i in ids]
        except Exception as e:
             print(f"Error getting all memories: {e}")
             return []

    def clear_semantic_memory(self):
        """Clear all semantic memory"""
        if not self.semantic_memory:
            return 0
            
        try:
            ids = self.semantic_memory.get().get("ids", [])
            count = len(ids)
            if count > 0:
                self.semantic_memory.delete(ids)
            print(f"     Semantic memory cleared ({count} conversations deleted)")
            return count
        except Exception as e:
            print(f"Error clearing memory: {e}")
            return 0
