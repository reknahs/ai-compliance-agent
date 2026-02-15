from typing import Dict


def step_7_store_conversation(state: Dict, memory_manager=None) -> Dict:
    """Step 7: Store conversation in memory after human approval"""
    print("\n Step 7: Storing Conversation in Memory")
    
    if not memory_manager or state.get("skip_memory") or not state.get("human_approved"):
        state["conversation_stored"] = False
        state["conversation_id"] = ""
        return state
    
    user_query = state["user_query"]
    structured_answer = state["structured_answer"]
    
    # short-term memory
    memory_manager.add_to_short_term(user_query, structured_answer)
    
    # semantic memory with metadata
    metadata = {
        "query_type": state.get("query_type", "unknown"),
        "citation_quality": state.get("citation_quality", "unknown"),
        "loop_count": state.get("loop_count", 0),
        "human_approved": True,
        "human_feedback": state.get("human_feedback", "")
    }
    print("storing")
    conv_id = memory_manager.store_conversation(
        user_query,
        structured_answer,
        metadata
    )
    
    state["conversation_stored"] = True
    state["conversation_id"] = conv_id
    
    stats = memory_manager.get_memory_stats()
    print(f"   Short-term memory: {stats['short_term_count']}/{stats['short_term_capacity']}")
    print(f"   Total conversations stored: {stats['semantic_memory_count']}")
    
    return state