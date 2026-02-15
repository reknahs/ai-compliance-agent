# AI Compliance & Security Agent 🛡️

A multi-step, memory-aware RAG agent designed to answer questions about AI compliance and security frameworks (NIST AI RMF, EU AI Act, ISO 42001, etc.).

Built with **LangGraph**, **LangChain**, and **Mem0** to provide context-aware, verifiable, and safe responses.

## 🚀 Key Features

- **🧠 Hybrid Memory System**: Integrates **Mem0** (for semantic user history) and a custom **ChromaDB** solution to recall past interactions and user preferences.
- **🔄 Multi-Step Reasoning**: Uses a graph-based workflow (Intent Analysis → Retrieval → Synthesis → Validation) to ensure high-quality answers.
- **✅ Self-Correction**: Automatically validates responses against retrieved evidence and loops back to refine answers if claims are unsupported.
- **👤 User Profiling**: extract and learns facts about the user (e.g., "I work in FinTech", "Our servers are in Germany") to tailor compliance advice.
- **🕵️ Fact Extraction**: Passively extracts and learns new information from conversations to build a dynamic knowledge base.
- **✋ Human-in-the-Loop**: Optional approval step ensures sensitive compliance advice is verified before delivery.

## 🛠️ Technology Stack

- **Framework**: [LangChain](https://www.langchain.com/) & [LangGraph](https://langchain-ai.github.io/langgraph/)
- **Memory**: [Mem0](https://mem0.ai/) & ChromaDB
- **LLMs**: Gemini (via Google GenAI) or Llama 3 (via Ollama)
- **Evaluation**: [DeepEval](https://github.com/confident-ai/deepeval)
- **Vector Store**: ChromaDB

## 📂 Project Structure

```
├── src/
│   └── compliance_agent/
│       ├── main.py              # Entry point & Agent Graph definition
│       ├── memory/              # Memory implementations (Mem0 & Custom)
│       ├── steps/               # Individual graph nodes (Analyze, Retrieve, Synthesize, etc.)
│       └── schemas/             # Pydantic models for structured output
├── data/
│   ├── memory_store/          # Local ChromaDB for custom memory
│   └── evaluation_results/    # Logs of performance tests
├── evals/                       # DeepEval test suite
└── requirements.txt             # Project dependencies
```

## ⚡ Setup & Installation

### 1. Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com/) (if using local LLMs)
- Google AI Studio API Key (if using Gemini)

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Config
Create a `.env` file in the root directory:
```bash
GOOGLE_API_KEY=your_google_api_key
MEMORY_API_KEY=your_mem0_api_key  # Optional if using custom memory
```

### 4. Ingest Documents (Required)
Before running the agent, you must ingest the compliance documents (PDFs) into the vector store:

1. Place your PDF documents (e.g., EU AI Act, NIST AI RMF) in `data/sources/`.
2. Run the ingestion script:
   ```bash
   python -m src.compliance_agent.ingestion
   ```
   This will chunk the documents and create a local ChromaDB vector store in `data/vector_store/`.

## 🏃 Usage

### Run the Agent
Start the interactive CLI:
```bash
python -m src.compliance_agent.main
```

### Options
- Use local custom memory (ChromaDB) instead of Mem0 Cloud:
  ```bash
  python -m src.compliance_agent.main --custom-memory
  ```
- Pass a query directly:
  ```bash
  python -m src.compliance_agent.main "What are the key requirements of the EU AI Act?"
  ```

## 🧪 Evaluation

Run the comprehensive test suite to measure RAG performance, Memory recall, and Fact Extraction accuracy:

```bash
python -m evals.run_evaluation
```

Key metrics tracked:
- **RAG**: Answer Relevancy, Faithfulness, Hallucination
- **Memory**: Recall accuracy of user details
- **Extraction**: Precision of fact extraction from conversation

---
*Built for the GenAI era of Compliance.* 
