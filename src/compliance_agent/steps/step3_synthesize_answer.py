from typing import Dict
import os
from langchain_core.messages import HumanMessage, SystemMessage

from ..state import get_llm


def step_3_synthesize_answer(state: Dict) -> Dict:
    """Step 3: Synthesizes answer with memory context"""
    print("\n   Step 3: Synthesizing Answer with Memory Context")
    
    llm = get_llm(temperature=0.3)
    user_query = state["user_query"]
    retrieved_chunks = state["retrieved_chunks"]
    
    recent_context = state.get("user_context", "No recent conversation.")
    relevant_memories = state.get("relevant_memories", "No relevant past conversations.")
    user_profile = state.get("user_profile", "No user profile available.")
    
    context = ""
    for i, chunk in enumerate(retrieved_chunks):
        source_name = os.path.basename(chunk["source"])
        score = chunk.get("relevance_score", 0.0)
        context += f"\n--- Source {i+1}: {source_name} (Page {chunk['page']}, Relevance: {score:.3f}) ---\n"
        context += chunk["content"]
        context += "\n"
    
    system_prompt = """You are an AI compliance and security expert with memory of past conversations. You provide accurate, well-cited answers using your knowledge base and conversation history.

CRITICAL CITATION RULES:
1. When you reference information from the compliance documents, cite it: [Source: document_name]
2. Use the EXACT document name from the context (e.g., NIST.AI.100-1.pdf, CELEX_32024R1689_EN_TXT.pdf)
3. If information is not in the provided documents, you can still answer based on general knowledge or memory, but make it clear
4. NEVER make up citations - only cite when directly referencing the provided documents

CONVERSATION STYLE:
1. Be natural and conversational, like a knowledgeable colleague
2. Don't force information into rigid templates or formats
3. Answer the question directly and naturally
4. Use your memory of past conversations to provide context and continuity
5. Reference the user's profile, role, and past discussions when relevant

PERSONALIZATION:
1. Adapt to the user's expertise level and preferences (check their profile)
2. If they prefer brief responses, be concise
3. If they prefer detailed explanations, elaborate
4. Maintain consistency with what you know about them

ANSWER STRATEGY:
1. Start by directly addressing the user's question
2. Draw from compliance documents when relevant (and cite them)
3. Use conversation history and memory to provide personalized context
4. Be helpful and conversational, not robotic or template-driven
5. If it's a general question (not about compliance), just answer naturally using memory"""

    user_prompt = f"""## What I Know About You
{user_profile}

## Recent Conversation
{recent_context}

## What We've Discussed Before
{relevant_memories}

## Compliance & Security Documents Available
{context if context.strip() else "No directly relevant compliance documents found for this query."}

## Your Question
"{user_query}"

Please answer naturally and conversationally. Reference documents when you use them, reference our past conversations when relevant, and be yourself - not a template-following robot."""

    try:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = llm.invoke(messages)
        state["structured_answer"] = response.content
        
    except Exception as e:
        print(f"   Error during synthesis: {e}")
        state["structured_answer"] = f"Error generating answer: {e}"
        
    answer = state["structured_answer"]
    citation_count = answer.count("[Source:")
    print(f"   Answer generated ({len(answer)} chars, {citation_count} citations)")
    
    return state