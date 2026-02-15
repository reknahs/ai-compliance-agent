import os
import time
from typing import List, Dict, Optional
from mem0 import MemoryClient

from .base import BaseMemoryManager


class Mem0MemoryManager(BaseMemoryManager):
    """Memory manager using Mem0 API."""
    
    def __init__(
        self,
        api_key: str = None,
        short_term_size: int = 10,
        profile_path: str = "data/memory_store/user_profile.json"
    ):
        super().__init__(short_term_size=short_term_size, profile_path=profile_path)
        
        # initialize Mem0 client
        if api_key is None:
            api_key = os.getenv("MEM0_API_KEY")
            if not api_key:
                print("     Mem0 API Key not provided. Mem0 operations may fail.")
        
        print(f"   Initializing Mem0 client")
        try:
            self.client = MemoryClient(api_key=api_key)
            print(f"     Mem0 client initialized")
        except Exception as e:
            print(f"     Failed to initialize Mem0 client: {e}")
            self.client = None

    def store_conversation(
        self,
        user_message: str,
        agent_response: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """Store conversation in Mem0"""
        if not self.client:
            return "failed_client_init"
            
        messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": agent_response}
        ]
        
        print(f"\n     Storing in Mem0")
        
        try:
            result = self.client.add(
                messages,
                user_id="default_user",
                metadata=metadata
            )
            
            print(f"     Stored in Mem0: {result}")
            
            time.sleep(2)
            
            return "stored"
            
        except Exception as e:
            print(f"     Error storing in Mem0: {e}")
            return "error"

    def retrieve_relevant_memories(
        self,
        query: str,
        k: int = 5
    ) -> List[Dict]:
        """Retrieve relevant memories using Mem0's search"""
        if not self.client:
            return []
            
        try:
            filters = {
                "OR": [
                    {"user_id": "default_user"}
                ]
            }
            
            results = self.client.search(
                query=query,
                user_id="default_user",
                filters=filters,
                limit=k
            )
        except Exception as e:
            print(f"     No memories found or error: {e}")
            return []
        
        memories = []
        for result in results.get("results", []):
            memory_text = result.get("memory", "")
            score = result.get("score", 0.0)
            result_metadata = result.get("metadata", {})
            
            memories.append({
                "memory_text": memory_text,
                "timestamp": result.get("created_at", ""),
                "conversation_id": result.get("id", ""),
                "relevance_score": score,
                "hybrid_score": score,
                "metadata": result_metadata
            })
        
        # sort by relevance score
        memories.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        return memories

    def get_all_memories(self) -> List[Dict]:
        """Get all memories from Mem0"""
        if not self.client:
            return []
            
        try:
            filters = {
                "OR": [
                    {"user_id": "default_user"}
                ]
            }
            all_memories = self.client.get_all(filters=filters)
            return all_memories.get("results", [])
        except Exception as e:
            print(f"     Error getting memories: {e}")
            return []

    def clear_semantic_memory(self):
        """Clear all semantic (long-term) memory"""
        if not self.client:
            return 0
            
        all_memories = self.get_all_memories()
        count = len(all_memories)
        
        for memory in all_memories:
            memory_id = memory.get("id")
            if memory_id:
                try:
                    self.client.delete(memory_id)
                except:
                    pass
        
        print(f"     Semantic memory cleared ({count} memories deleted)")
        return count
