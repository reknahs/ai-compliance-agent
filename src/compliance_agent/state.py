import os
from typing import TypedDict, List, Dict, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.globals import set_llm_cache
from langchain_core.caches import InMemoryCache
from langchain_ollama import ChatOllama

# load environment variables
load_dotenv()

# enable LLM caching for faster repeated queries
set_llm_cache(InMemoryCache())

PERSIST_DIRECTORY = os.path.join("data", "vector_store")
EMBEDDING_MODEL = "mixedbread-ai/mxbai-embed-large-v1"


class AgentState(TypedDict):
    """State that gets passed between agent steps."""
    # input
    user_query: str
    
    # Step 1: Intent Analysis + User Context Retrieval
    intent_analysis: str
    query_type: str
    missing_context: List[str]
    user_context: str
    
    # Step 2: Retrieval
    retrieved_chunks: List[Dict]
    retrieval_scores: List[float]
    relevant_memories: str  
    
    # Step 3: Synthesis
    structured_answer: str
    
    # Step 4: Validation
    validation_notes: str
    validation_decision: str
    citation_quality: str
    unsupported_claims: List[str]
    
    # Step 5: Follow-up
    follow_up_questions: List[str]
    
    # Step 6: Human Approval
    human_approved: bool
    auto_approve: bool

    # Step 7: Store Conversation
    conversation_stored: bool
    conversation_id: str
    skip_memory: bool

    # Step 8: Fact Extraction & Profile
    user_profile: str  
    extracted_facts: List[Dict] 
    profile_updated: bool  
    profile_conflicts: List[str]  
    
    # Final output
    final_response: str
    
    # Conditional fields
    loop_count: int
    previous_citation_quality: str
    loop_reason: str


def get_llm(temperature: float = 0.3):
    """Initialize LLM"""
    
    try:
        model_name = os.getenv("LOCAL_LLM_MODEL", "llama3.2:latest")
        print(f"   Using Ollama with model: {model_name}")
        
        return ChatOllama(
            model=model_name,
            temperature=temperature,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            num_ctx=8192, 
            num_predict=1024,
            repeat_penalty=1.1,
        )
    except Exception as e:
        print(f"   Ollama not available or model not found: {e}")


def get_structured_llm(schema: type[BaseModel], temperature: float = 0.3):
    """ Get LLM with structured output enforcement using Pydantic schema."""
    try:
        model_name = os.getenv("LOCAL_LLM_MODEL", "llama3.2:latest")
        
        llm = ChatOllama(
            model=model_name,
            temperature=temperature,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            num_ctx=8192,
            num_predict=1024,
            repeat_penalty=1.1,
            format="json", 
        )
        return llm.with_structured_output(schema)
    except Exception as e:
        print(f"   Ollama structured output failed: {e}")


def get_vector_store():
    """Initialize vector store"""
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    print(f"   Using embedding model: {EMBEDDING_MODEL}")
    
    if not os.path.exists(PERSIST_DIRECTORY):
        raise ValueError(
            f"Vector store not found at '{PERSIST_DIRECTORY}'. "
            "Please run 'python ingest.py' first to create the vector store."
        )
    
    return Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings
    )


def get_retriever_with_scores(vector_store, k: int = 5):
    """Return retriever for relevance scores."""
    def retrieve(query: str):
        results = vector_store.similarity_search_with_relevance_scores(query, k=k)
        docs = [doc for doc, _ in results]
        scores = [score for _, score in results]
        return docs, scores
    
    return retrieve