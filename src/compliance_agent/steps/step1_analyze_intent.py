from typing import Dict
from langchain_core.messages import HumanMessage, SystemMessage

from ..state import get_llm, get_structured_llm
from ..schemas.workflow import IntentAnalysis


def step_1_analyze_intent(state: Dict, memory_manager=None) -> Dict:
    """Step 1: Uses structured output and retrieves user context from memory"""
    print("\nStep 1: Analyzing Intent + Retrieving User Context")
    
    user_query = state["user_query"]
    
    # retrieve user context from short-term memory
    if memory_manager:
        state["user_context"] = memory_manager.get_short_term_context()
        print(f"   Retrieved {len(memory_manager.short_term_memory)} recent messages")
        
        # retrieve user profile
        state["user_profile"] = memory_manager.profile_manager.get_relevant_profile_info(user_query)
        print(f"   Retrieved user profile")
    else:
        state["user_context"] = "No conversation history available."
        state["user_profile"] = "No user profile available."
    
    is_loop_back = state.get("loop_count", 0) > 0 and state.get("validation_decision") == "loop_to_intent"
    
    llm = get_structured_llm(IntentAnalysis, temperature=0.2)
    
    # few-shot prompting with examples and memory context
    if is_loop_back:
        validation_notes = state.get("validation_notes", "")
        loop_reason = state.get("loop_reason", "")
        unsupported_claims = state.get("unsupported_claims", [])
        
        system_message = """You are an AI compliance and security expert. Analyze user queries to understand intent and identify missing context.

Based on validation feedback, provide MORE SPECIFIC analysis focusing on gaps identified."""

        user_message = f"""USER QUERY: "{user_query}"

USER PROFILE:
{state.get("user_profile", "No profile")}

CONVERSATION CONTEXT:
{state.get("user_context", "No context")}

PREVIOUS VALIDATION FEEDBACK:
{validation_notes}

LOOP REASON: {loop_reason}

UNSUPPORTED CLAIMS FROM PREVIOUS ATTEMPT:
{unsupported_claims if unsupported_claims else "None"}

Provide a more targeted analysis that addresses these specific gaps. Be very specific about what context is missing (e.g., "deployment region for EU AI Act compliance", "types of user data processed for privacy assessment", "industry sector for regulatory requirements").

Analyze and return structured response."""
    else:
        system_message = """You are an AI compliance and security expert with memory of past conversations. Analyze user queries to understand intent and identify missing context.

IMPORTANT CLASSIFICATION RULES:
- "What is..." or "Define..." → query_type = "definition"
- "What risks..." or "security concerns" → query_type = "security_risk"  
- "What compliance..." or "regulatory requirements" → query_type = "compliance"
- "Compare..." or "difference between" → query_type = "comparison"
- Everything else → query_type = "general"

MEMORY AWARENESS:
- Consider conversation history when analyzing intent
- If user refers to "my project" or "our system", check conversation history for context
- Identify if this is a follow-up to a previous question

EXAMPLES:

Query: "What is the NIST AI Risk Management Framework?"
→ query_type: "definition"
→ missing_context: [] (sufficient for definition)

Query: "What security risks apply to an AI chatbot?"
→ query_type: "security_risk"
→ missing_context: ["type of data handled", "deployment environment", "user base"]

Query: "What EU AI Act compliance is needed?"
→ query_type: "compliance"
→ missing_context: ["AI system purpose", "risk level", "deployment region"]

Query: "Compare NIST AI RMF and EU AI Act"
→ query_type: "comparison"
→ missing_context: [] (sufficient for high-level comparison)"""

        user_message = f"""USER PROFILE:
{state.get("user_profile", "No profile information")}

CONVERSATION CONTEXT:
{state.get("user_context", "No conversation history")}

USER QUERY: "{user_query}"

Analyze this query considering the user profile and conversation history. Return structured response with:
- intent_analysis: Brief statement of what user is asking (reference profile/past conversation if relevant)
- query_type: One of [security_risk, compliance, comparison, definition, general]
- missing_context: List of specific missing details, or empty list if query is sufficient"""

    # structured output with error handling
    try:
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=user_message)
        ]
        
        result: IntentAnalysis = llm.invoke(messages)
        
        state["intent_analysis"] = result.intent_analysis
        state["query_type"] = result.query_type
        state["missing_context"] = result.missing_context
        
    except Exception as e:
        print(f"   Structured output failed, using fallback: {e}")
        fallback_llm = get_llm(temperature=0.2)
        response = fallback_llm.invoke([
            SystemMessage(content=system_message),
            HumanMessage(content=user_message)
        ])
        
        content = response.content
        state["intent_analysis"] = content.split("intent_analysis:")[-1].split("\n")[0].strip() if "intent_analysis:" in content else "Query analysis"
        state["query_type"] = "general"
        state["missing_context"] = []
    
    print(f"   Intent: {state['intent_analysis']}")
    print(f"   Query Type: {state['query_type']}")
    print(f"   Missing Context: {state['missing_context'] if state['missing_context'] else 'None'}")
    
    return state