from typing import Dict
import os

from ..state import get_vector_store


def step_2_retrieve_documents(state: Dict, memory_manager=None) -> Dict:
    """Step 2: Retrieves  documents and relevant memories"""
    print("\nStep 2: Retrieving Relevant Documents + Memories")
    
    vector_store = get_vector_store()
    user_query = state["user_query"]
    
    is_loop_back = state.get("loop_count", 0) > 0 and state.get("validation_decision") == "loop_to_retrieval"
    
    # k values
    k = 12
    fetch_k = 40
    
    # enhanced query with validation feedback
    enhanced_query = user_query
    if is_loop_back:
        validation_notes = state.get("validation_notes", "")
        unsupported_claims = state.get("unsupported_claims", [])
        
        # add unsupported claims to query to find missing evidence
        if unsupported_claims:
            enhanced_query = f"{user_query} {' '.join(unsupported_claims[:2])}"
        else:
            enhanced_query = f"{user_query} {validation_notes}"
        
        print(f"   Enhanced retrieval (Loop {state['loop_count']}) addressing: {state.get('loop_reason', '')}")
    
    # use MMR search for diversity (reduces redundancy)
    lambda_mult = 0.6
    
    try:
        print(f"   Using MMR search (k={k}, fetch_k={fetch_k}, lambda={lambda_mult})")
        results = vector_store.max_marginal_relevance_search(
            enhanced_query,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult
        )
        
        scored_results = vector_store.similarity_search_with_relevance_scores(enhanced_query, k=k)
        scores = [score for doc, score in scored_results]
        
    except Exception as e:
        print(f"   MMR search failed ({e}), falling back to similarity search")
        scored_results = vector_store.similarity_search_with_relevance_scores(enhanced_query, k=k)
        results = [doc for doc, score in scored_results]
        scores = [score for doc, score in scored_results]
    
    # Track relevance scores and filter low-quality results
    retrieved_chunks = []
    min_score_threshold = 0.3
    
    for i, (doc, score) in enumerate(zip(results, scores)):
        if score < min_score_threshold:
            continue
            
        chunk_info = {
            "content": doc.page_content,
            "source": doc.metadata.get("source", "Unknown"),
            "page": doc.metadata.get("page", "N/A"),
            "rank": i + 1,
            "relevance_score": score
        }
        retrieved_chunks.append(chunk_info)
    
    state["retrieved_chunks"] = retrieved_chunks
    state["retrieval_scores"] = scores[:len(retrieved_chunks)]
    
    # retrieve relevant memories from semantic memory
    if memory_manager:
        print(f"   Retrieving relevant memories")
        relevant_memories = memory_manager.get_relevant_memories_context(user_query, k=5)
        state["relevant_memories"] = relevant_memories
        
    else:
        state["relevant_memories"] = "No memory system available."
        
    # show source distribution
    sources = {}
    for chunk in retrieved_chunks:
        source_name = os.path.basename(chunk["source"])
        sources[source_name] = sources.get(source_name, 0) + 1
    
    for source, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
        print(f"     - {source}: {count} chunks")
    
    return state