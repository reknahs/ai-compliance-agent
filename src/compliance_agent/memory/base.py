import os
from typing import List, Dict, Optional, Any
from collections import deque
from datetime import datetime
from abc import ABC, abstractmethod

from ..user_profile import UserProfileManager


class BaseMemoryManager(ABC):
    def __init__(
        self,
        short_term_size: int = 10,
        profile_path: str = "user_profile.json"
    ):
        self.short_term_size = short_term_size
        
        # short-term memory (in-memory deque)
        self.short_term_memory = deque(maxlen=short_term_size)
        
        # flag to disable short-term memory (for testing)
        self.disable_short_term = False
        
        self.profile_manager = UserProfileManager(profile_path=profile_path)
        print(f"   User profile initialized at {profile_path}")

    def add_to_short_term(self, user_message: str, agent_response: str):
        """Add a message pair to short-term memory"""
        self.short_term_memory.append({
            "user": user_message,
            "agent": agent_response,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_short_term_context(self, skip_if_disabled: bool = True) -> str:
        """Get formatted short-term memory context"""
        if skip_if_disabled and self.disable_short_term:
            return "No recent conversation history."
        
        if not self.short_term_memory:
            return "No recent conversation history."
        
        context = "## Recent Conversation History\n\n"
        for i, msg in enumerate(self.short_term_memory, 1):
            context += f"**Turn {i}:**\n"
            context += f"User: {msg['user']}\n"
            context += f"Agent: {msg['agent'][:200]}{'...' if len(msg['agent']) > 200 else ''}\n\n"
        
        return context
    
    def clear_short_term(self):
        """Clear short-term memory"""
        self.short_term_memory.clear()
        print("     Short-term memory cleared")

    @abstractmethod
    def store_conversation(
        self,
        user_message: str,
        agent_response: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """Store conversation in semantic memory"""
        pass

    @abstractmethod
    def retrieve_relevant_memories(
        self,
        query: str,
        k: int = 5
    ) -> List[Dict]:
        """Retrieve relevant memories"""
        pass

    @abstractmethod
    def get_all_memories(self) -> List[Dict]:
        """Get all memories"""
        pass

    @abstractmethod
    def clear_semantic_memory(self):
        """Clear all semantic memory"""
        pass

    def get_relevant_memories_context(self, query: str, k: int = 5) -> str:
        """Get formatted context from extracted memories"""
        memories = self.retrieve_relevant_memories(query, k=k)
        
        if not memories:
            # fall back to short-term if not in testing mode
            if not self.disable_short_term and self.short_term_memory:
                print("     No semantic memories found, falling back to short-term memory")
                return self.get_short_term_context(skip_if_disabled=False)
            
            print("     No relevant past memories found")
            return "No relevant past information found."
        
        context = "## What I Remember\n\n"
        
        print(f"\n     Retrieved {len(memories)} relevant past memories:")
        print("   " + "=" * 76)
        for i, mem in enumerate(memories, 1):
            timestamp = mem.get("timestamp", "unknown")
            if timestamp and len(timestamp) > 10:
                timestamp = timestamp[:10]
            
            # get content from various possible keys
            memory_text = mem.get("memory_text", "")
            if not memory_text:
                if "user_message" in mem:
                     memory_text = f"User: {mem['user_message']} | Agent: {mem.get('agent_response', '')}"
                elif "memory" in mem:
                     memory_text = mem["memory"]
                else:
                     memory_text = "No content"

            relevance = mem.get("relevance_score", 0.0)
            hybrid_score = mem.get("hybrid_score", relevance)
            
            print(f"   {i}. [{hybrid_score:.2f}] {memory_text[:100]}")
            print(f"      Date: {timestamp}")
            print()
            
            context += f"{i}. {memory_text}"
            context += f" _(from {timestamp}, relevance: {hybrid_score:.2f})_\n"
        
        print("   " + "=" * 76)
        
        return context

    def get_memory_stats(self) -> Dict:
        """Get statistics about memory usage"""
        try:
            semantic_count = len(self.get_all_memories())
        except Exception as e:
            print(f"Error getting memory stats: {e}")
            semantic_count = 0
        
        # get profile stats
        try:
            profile_stats = self.profile_manager.get_profile_stats()
            
            personal_info_count = 1 if any([
                self.profile_manager.profile.personal_info.name,
                self.profile_manager.profile.personal_info.role,
                self.profile_manager.profile.personal_info.company,
                self.profile_manager.profile.personal_info.location,
                self.profile_manager.profile.personal_info.industry
            ]) else 0
            
            profile_facts = (
                profile_stats["preference_count"] + 
                profile_stats["expertise_count"] +
                personal_info_count
            )
            profile_last_updated = profile_stats["last_updated"]
        except Exception:
            profile_facts = 0
            profile_last_updated = "unknown"
        
        return {
            "short_term_count": len(self.short_term_memory),
            "short_term_capacity": self.short_term_size,
            "semantic_memory_count": semantic_count,
            "profile_facts": profile_facts,
            "profile_last_updated": profile_last_updated
        }
