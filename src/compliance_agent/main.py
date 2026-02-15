import os
import sys
import argparse
from typing import Dict
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END

from .state import AgentState
from .memory import Mem0MemoryManager, CustomMemoryManager
from .steps import (
    step_1_analyze_intent,
    step_2_retrieve_documents,
    step_3_synthesize_answer,
    step_4_validate_response,
    step_5_generate_followups,
    step_6_human_approval,
    step_7_store_conversation,
    step_8_extract_facts
)

# load environment variables
load_dotenv()

# global memory manager instance
MEMORY_MANAGER = None


def get_memory_manager(use_custom_memory: bool = False):
    """Get or create global memory manager instance"""
    global MEMORY_MANAGER
    
    if MEMORY_MANAGER is not None:
        return MEMORY_MANAGER

    print("\nInitializing Memory System")
    
    if use_custom_memory:
        MEMORY_MANAGER = CustomMemoryManager(
            memory_db_path=os.path.join("data", "memory_store"),
            profile_path=os.path.join("data", "memory_store", "user_profile.json")
        )
    else:
        api_key = os.getenv("MEMORY_API_KEY")
        MEMORY_MANAGER = Mem0MemoryManager(
            api_key=api_key,
            short_term_size=10,
            profile_path=os.path.join("data", "memory_store", "user_profile.json")
        )
        
    stats = MEMORY_MANAGER.get_memory_stats()
    print(f"   Memory initialized: {stats['semantic_memory_count']} conversations in long-term memory")
    return MEMORY_MANAGER


def create_agent_graph(use_custom_memory: bool = False):
    """ Create the multi-step agent with memory and human approval"""
    memory_manager = get_memory_manager(use_custom_memory=use_custom_memory)
    
    workflow = StateGraph(AgentState)
    
    # add nodes with error wrapping and memory manager injection
    workflow.add_node("analyze_intent", _wrap_with_error_handling(
        lambda state: step_1_analyze_intent(state, memory_manager), "Intent Analysis"
    ))
    workflow.add_node("retrieve_documents", _wrap_with_error_handling(
        lambda state: step_2_retrieve_documents(state, memory_manager), "Document Retrieval"
    ))
    workflow.add_node("synthesize_answer", _wrap_with_error_handling(
        step_3_synthesize_answer, "Answer Synthesis"
    ))
    workflow.add_node("validate_response", _wrap_with_error_handling(
        step_4_validate_response, "Response Validation"
    ))
    workflow.add_node("generate_followups", _wrap_with_error_handling(
        step_5_generate_followups, "Follow-up Generation"
    ))
    workflow.add_node("human_approval", _wrap_with_error_handling(
        step_6_human_approval, "Human Approval"
    ))
    workflow.add_node("store_conversation", _wrap_with_error_handling(
        lambda state: step_7_store_conversation(state, memory_manager), "Store Conversation"
    ))
    workflow.add_node("extract_facts", _wrap_with_error_handling(
        lambda state: step_8_extract_facts(state, memory_manager), "Extract Facts"
    ))
    
    # define flow
    workflow.set_entry_point("analyze_intent")
    workflow.add_edge("analyze_intent", "retrieve_documents")
    workflow.add_edge("retrieve_documents", "synthesize_answer")
    workflow.add_edge("synthesize_answer", "validate_response")
    
    # conditional routing from validation
    workflow.add_conditional_edges(
        "validate_response",
        route_after_validation,
        {
            "continue": "generate_followups",
            "loop_to_intent": "analyze_intent",
            "loop_to_retrieval": "retrieve_documents",
            "max_loops_reached": "generate_followups"
        }
    )
    
    # add human approval, storage, and fact extraction steps
    workflow.add_edge("generate_followups", "human_approval")
    workflow.add_edge("human_approval", "store_conversation")
    workflow.add_edge("store_conversation", "extract_facts")
    workflow.add_edge("extract_facts", END)
    
    return workflow.compile()


def _wrap_with_error_handling(step_func, step_name: str):
    """Wrap step functions with error handling"""
    def wrapped(state: Dict) -> Dict:
        try:
            return step_func(state)
        except Exception as e:
            print(f"\nError in {step_name}: {e}")
            import traceback
            traceback.print_exc()
            
            if "intermediate_steps" not in state:
                state["intermediate_steps"] = []
            state["intermediate_steps"].append(f"Error in {step_name}: {str(e)[:100]}")
            
            if step_name == "Response Validation":
                state["validation_decision"] = "continue"
                state["citation_quality"] = "Unknown"
                state["validation_notes"] = f"Validation failed: {e}"
            
            if step_name == "Human Approval":
                state["human_approved"] = False
                state["human_feedback"] = f"Error during approval: {e}"
            
            return state
    
    return wrapped


def route_after_validation(state: Dict) -> str:
    """Conditional routing logic after validation step"""
    if "loop_count" not in state:
        state["loop_count"] = 0
        state["previous_citation_quality"] = ""
    
    decision = state.get("validation_decision", "continue")
    
    if decision in ["loop_to_intent", "loop_to_retrieval"]:
        state["loop_count"] += 1
        
        if state["loop_count"] > 2:
            print(f"\n  Maximum loops ({state['loop_count']}) exceeded. Continuing to final steps.")
            return "max_loops_reached"
        
        if decision == "loop_to_intent":
            print(f"\n  Looping back to intent analysis (Loop {state['loop_count']}/2)")
            print(f"   Reason: {state.get('loop_reason', 'Unknown')}")
            return "loop_to_intent"
        else:  
            print(f"\n  Looping back to document retrieval (Loop {state['loop_count']}/2)")
            print(f"   Reason: {state.get('loop_reason', 'Unknown')}")
            return "loop_to_retrieval"
    else:
        print(f"\n  Validation passed. Continuing to final steps.")
        return "continue"


def query_agent(
    user_query: str, 
    verbose: bool = True, 
    auto_approve: bool = False, 
    skip_memory: bool = False,
    use_custom_memory: bool = False
) -> Dict:
    """Query the agent with memory and human approval"""
    print("\n" + "=" * 80)
    print("AI COMPLIANCE & SECURITY AGENT with MEMORY (IMPROVED)")
    print("=" * 80)
    print(f"\n  User Query: {user_query}\n")
    
    if not user_query or len(user_query.strip()) < 5:
        raise ValueError("Query must be at least 5 characters long")
    
    initial_state = {
        "user_query": user_query,
        "intent_analysis": "",
        "query_type": "",
        "missing_context": [],
        "user_context": "",
        "user_profile": "",
        "retrieved_chunks": [],
        "retrieval_scores": [],
        "relevant_memories": "",
        "structured_answer": "",
        "validation_notes": "",
        "citation_quality": "",
        "unsupported_claims": [],
        "follow_up_questions": [],
        "human_approved": False,
        "human_feedback": "",
        "conversation_stored": False,
        "conversation_id": "",
        "extracted_facts": [],
        "profile_updated": False,
        "profile_conflicts": [],
        "final_response": "",
        "intermediate_steps": [],
        "validation_decision": "continue",
        "loop_count": 0,
        "previous_citation_quality": "",
        "loop_reason": "",
        "auto_approve": auto_approve,
        "skip_memory": skip_memory
    }
    
    # create agent
    agent = create_agent_graph(use_custom_memory=use_custom_memory)
    
    try:
        final_state = initial_state.copy()  
        for event in agent.stream(initial_state):
            for node_name, node_output in event.items():
                if isinstance(node_output, dict):
                    final_state.update(node_output)
        
        if verbose:
            print("\n" + "=" * 80)
            print("AGENT EXECUTION COMPLETE")
            print("=" * 80)
            print("\n  Execution Summary:")
            print(f"  - Query Type: {final_state['query_type']}")
            print(f"  - Citation Quality: {final_state['citation_quality']}")
            print(f"  - Documents Retrieved: {len(final_state['retrieved_chunks'])}")
            print(f"  - Refinement Loops: {final_state['loop_count']}")
            print(f"  - Human Approved: {final_state.get('human_approved', False)}")
            print(f"  - Stored in Memory: {final_state.get('conversation_stored', False)}")
            print(f"\n  Intermediate Steps:")
            for step in final_state["intermediate_steps"]:
                print(f"  {step}")
        
        return {
            "response": final_state["final_response"],
            "query_type": final_state["query_type"],
            "citation_quality": final_state["citation_quality"],
            "follow_up_questions": final_state["follow_up_questions"],
            "intermediate_steps": final_state["intermediate_steps"],
            "retrieval_scores": final_state.get("retrieval_scores", []),
            "loop_count": final_state.get("loop_count", 0),
            "unsupported_claims": final_state.get("unsupported_claims", []),
            "human_approved": final_state.get("human_approved", False),
            "conversation_stored": final_state.get("conversation_stored", False),
            "conversation_id": final_state.get("conversation_id", ""),
            "extracted_facts": final_state.get("extracted_facts", [])
        }
    
    except Exception as e:
        print(f"\n  Error during agent execution: {e}")
        import traceback
        traceback.print_exc()
        raise


def main():
    """Main CLI with memory system and conversation loop"""
    print("\n" + "=" * 80)
    print("AI COMPLIANCE & SECURITY AGENT with MEMORY (IMPROVED)")
    print("Multi-Step RAG Agent with Memory & Human-in-the-Loop")
    print("=" * 80)
    
    parser = argparse.ArgumentParser(description="AI Compliance Agent")
    parser.add_argument("--custom-memory", action="store_true", help="Use local custom memory (ChromaDB) instead of Mem0")
    parser.add_argument("query", nargs="*", help="Initial query to run")
    args = parser.parse_args()
    
    use_custom_memory = args.custom_memory
    
    memory_manager = get_memory_manager(use_custom_memory=use_custom_memory)
    stats = memory_manager.get_memory_stats()
    print(f"\n  Memory Status:")
    print(f"   - Short-term: {stats['short_term_count']}/{stats['short_term_capacity']} messages")
    print(f"   - Long-term: {stats['semantic_memory_count']} conversations stored")
    print(f"   - Profile: {stats['profile_facts']} facts learned")
    
    extra_args_query = " ".join(args.query) if args.query else ""
    first_run = True
    
    while True:
        if first_run and extra_args_query and stats['short_term_count'] == 0:
            user_query = extra_args_query
            first_run = False
        else:
            print()
            user_query = input("Enter your question: ").strip()
        
        if not user_query or user_query.lower() in ['exit', 'quit', 'q']:
            print("\n  Ending conversation. Goodbye!")
            break
                
        try:
            result = query_agent(user_query, verbose=True)
            
            print("\n" + "=" * 80)
            print("  RESPONSE METRICS")
            print("=" * 80)
            print(f"Query Type: {result['query_type']}")
            print(f"Citation Quality: {result['citation_quality']}")
            print(f"Refinement Loops: {result['loop_count']}")
            print(f"Human Approved: {result['human_approved']}")
            print(f"Stored in Memory: {result['conversation_stored']}")
            
            if result['retrieval_scores']:
                avg_score = sum(result['retrieval_scores']) / len(result['retrieval_scores'])
                print(f"Average Retrieval Relevance: {avg_score:.1%}")
            
            if result['unsupported_claims']:
                print(f"\n  Unsupported Claims: {len(result['unsupported_claims'])}")
            
            stats = memory_manager.get_memory_stats()
            print(f"\n  Updated Memory: {stats['short_term_count']}/{stats['short_term_capacity']} recent messages, {stats['profile_facts']} profile facts")
            
        except Exception as e:
            print(f"\n  Error: {e}")
            import traceback
            traceback.print_exc()
            print("\nContinuing to next question")

if __name__ == "__main__":
    # Clear Mem0 and Custom Memory
    # mem0_manager = get_memory_manager(use_custom_memory=False)
    # mem0_manager.clear_semantic_memory()
    # mem0_manager.profile_manager.clear_profile()
    # MEMORY_MANAGER = None
    # custom_manager = get_memory_manager(use_custom_memory=True)
    # custom_manager.clear_semantic_memory()    

    main()