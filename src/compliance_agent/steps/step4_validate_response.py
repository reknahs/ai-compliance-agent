from typing import Dict
from langchain_core.messages import HumanMessage, SystemMessage

from ..state import get_structured_llm
from ..schemas.workflow import ValidationResult


def step_4_validate_response(state: Dict) -> Dict:
    """Step 4: Uses structured validation with per-claim tracking"""
    print("\nStep 4: Validating Response")
    
    structured_answer = state["structured_answer"]
    retrieved_chunks = state["retrieved_chunks"]
    
    try:
        llm = get_structured_llm(ValidationResult, temperature=0.1)
        
        system_prompt = """You are a quality assurance expert for AI-generated responses.

VALIDATION CRITERIA:

**Citation Quality Levels:**
- Excellent: When referencing compliance documents, 90%+ of claims are cited with sources
- Good: When referencing compliance documents, 70-89% of claims are cited  
- Fair: Some document claims cited, but missing several important citations
- Poor: Makes claims about compliance documents without citing them

**IMPORTANT CONTEXT:**
1. Not all queries require compliance document citations (e.g., "what did we discuss?", "tell me about yourself")
2. The agent can answer from memory and general knowledge WITHOUT citations - that's fine
3. ONLY check citation quality when the answer actually references specific compliance documents
4. If the query is conversational/memory-based and doesn't reference documents, rate as "Good"

**What to Check:**
1. Does the answer reference specific information from compliance documents?
2. If yes, are those references properly cited with [Source: document_name]?
3. Is the answer helpful and accurate based on the query type?
4. For comparison queries: Are BOTH frameworks cited if both are discussed?

**Unsupported Claims:**
Only flag claims that reference compliance documents but lack citations."""

        user_prompt = f"""ANSWER TO VALIDATE:
{structured_answer}

NUMBER OF COMPLIANCE SOURCES PROVIDED: {len(retrieved_chunks)}
QUERY TYPE: {state.get('query_type', 'unknown')}
USER QUERY: "{state.get('user_query', '')}"

Validate this answer:
1. Is this a compliance-related query that requires document citations?
2. If the answer references compliance documents, are they cited properly?
3. If it's a conversational query (about memory, personal questions), is it helpful?
4. What's the overall quality?"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        result: ValidationResult = llm.invoke(messages)
        
        state["citation_quality"] = result.citation_quality
        state["validation_notes"] = result.validation_notes
        state["unsupported_claims"] = result.unsupported_claims[:5] 
        
    except Exception as e:
        print(f"   Structured validation failed, using fallback: {e}")
        citation_count = structured_answer.count("[Source:")
        answer_length = len(structured_answer)
        user_query = state.get("user_query", "").lower()
        
        conversational_keywords = ["what did we", "tell me about", "do you remember", "what do you know", "our conversation"]
        is_conversational = any(keyword in user_query for keyword in conversational_keywords)
        
        is_json_output = structured_answer.strip().startswith("{") or "\"title\":" in structured_answer
        
        if is_json_output:
            state["citation_quality"] = "Poor"
            state["validation_notes"] = "Answer is in JSON format instead of formatted text - needs regeneration"
            state["unsupported_claims"] = ["Entire answer is JSON formatted"]
        elif is_conversational and answer_length > 100:
            state["citation_quality"] = "Good"
            state["validation_notes"] = f"Conversational query answered from memory (no document citations needed)"
            state["unsupported_claims"] = []
        elif no_documents and answer_length > 100:
            state["citation_quality"] = "Good"
            state["validation_notes"] = "No compliance documents relevant to query, answered from general knowledge/memory"
            state["unsupported_claims"] = []
        elif citation_count == 0 and len(retrieved_chunks) > 0:
            state["citation_quality"] = "Poor"
            state["validation_notes"] = f"Documents available but not cited in answer"
            state["unsupported_claims"] = ["Answer should reference available compliance documents"]
        elif citation_count < 3 and len(retrieved_chunks) > 5:
            state["citation_quality"] = "Fair"
            state["validation_notes"] = f"Only {citation_count} citations, could reference more available documents"
            state["unsupported_claims"] = [f"Could add more citations from {len(retrieved_chunks)} available sources"]
        elif citation_count < 5:
            state["citation_quality"] = "Good"
            state["validation_notes"] = f"Found {citation_count} citations, adequate coverage"
            state["unsupported_claims"] = []
        else:
            state["citation_quality"] = "Excellent"
            state["validation_notes"] = f"Found {citation_count} citations, comprehensive coverage"
            state["unsupported_claims"] = []
    
    print(f"   Citation Quality: {state['citation_quality']}")
    print(f"   Notes: {state['validation_notes']}")
    
    if state["unsupported_claims"]:
        print(f"   Unsupported Claims ({len(state['unsupported_claims'])}):")
        for i, claim in enumerate(state["unsupported_claims"][:3], 1):
            print(f"     {i}. {claim[:80]}")
    
    if "loop_count" not in state:
        state["loop_count"] = 0
        state["previous_citation_quality"] = ""
    
    citation_quality = state["citation_quality"]
    previous_quality = state.get("previous_citation_quality", "")
    
    quality_order = {"Poor": 0, "Fair": 1, "Good": 2, "Excellent": 3}
    current_score = quality_order.get(citation_quality, 0)
    previous_score = quality_order.get(previous_quality, 0)
    
    improved = current_score > previous_score if previous_quality else False
    
    no_documents = len(state.get("retrieved_chunks", [])) == 0

    same_quality_as_before = citation_quality == previous_quality and previous_quality != ""
    
    # routing decision
    if citation_quality in ["Poor", "Fair"]:
        # if no documents found, this is likely not a compliance query, don't loop
        if no_documents:
            state["validation_decision"] = "continue"
            state["loop_reason"] = "No compliance documents found - query may not be compliance-related"
        # if quality is the same as previous loop, we're stuck, stop looping
        elif same_quality_as_before:
            state["validation_decision"] = "continue"
            state["loop_reason"] = "Quality not improving - stopping to prevent infinite loop"
        # check if we're improving (don't loop if getting worse)
        elif state["loop_count"] > 0 and not improved:
            state["validation_decision"] = "continue"
            state["loop_reason"] = "Quality degraded from previous attempt"
        else:
            missing_info = state.get("validation_notes", "")
            has_unsupported = len(state.get("unsupported_claims", [])) > 0
            
            if "missing" in missing_info.lower() or not has_unsupported:
                state["validation_decision"] = "loop_to_intent"
                state["loop_reason"] = f"Missing context: {missing_info}"
                print(f"     Decision: Loop to intent (missing context)")
            else:
                state["validation_decision"] = "loop_to_retrieval"
                state["loop_reason"] = f"Need better sources for unsupported claims"
                print(f"     Decision: Loop to retrieval (unsupported claims)")
    else:
        state["validation_decision"] = "continue"
        state["loop_reason"] = "Validation passed"
        print(f"     Decision: Continue to final steps")
    
    state["previous_citation_quality"] = citation_quality
    
    return state