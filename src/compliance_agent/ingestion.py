import os
import glob
import shutil

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_experimental.text_splitter import SemanticChunker

SOURCE_DIRECTORY = os.path.join("data", "sources")
PERSIST_DIRECTORY = os.path.join("data", "vector_store")

EMBEDDING_MODEL = "mixedbread-ai/mxbai-embed-large-v1"


def get_embeddings():
    """Load embedding model with fallback."""
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    print(f"   Using embedding model: {EMBEDDING_MODEL}")
    return embeddings


def ingest_documents():
    """Reads PDFs with semantic chunking and better embeddings."""
    
    pdf_files = glob.glob(os.path.join(SOURCE_DIRECTORY, "*.pdf"))

    print(f"Found {len(pdf_files)} PDFs: {[os.path.basename(f) for f in pdf_files]} in '{SOURCE_DIRECTORY}'")

    # load embeddings
    embeddings = get_embeddings()

    all_splits = []
    
    print(f"\n  Using Semantic Chunking (splits at semantic boundaries)")
    text_splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",  
        breakpoint_threshold_amount=95 
    )

    # process PDFs
    for pdf_path in pdf_files:
        try:
            print(f"   Processing: {os.path.basename(pdf_path)}")
            loader = PyPDFLoader(pdf_path)
            raw_docs = loader.load()
            
            # split the documents into semantically meaningful chunks
            splits = text_splitter.split_documents(raw_docs)
            all_splits.extend(splits)
            
            # calculate average chunk size
            avg_size = sum(len(s.page_content) for s in splits) / len(splits) if splits else 0
            print(f"     Created {len(splits)} chunks (avg size: {avg_size:.0f} chars)")
            
        except Exception as e:
            print(f"Error processing {os.path.basename(pdf_path)}: {e}")

    print(f"\nTotal chunks created: {len(all_splits)}")
    
    # calculate statistics
    chunk_sizes = [len(s.page_content) for s in all_splits]
    print(f"   Avg chunk size: {sum(chunk_sizes) / len(chunk_sizes):.0f} chars")

    # clear old vector store if it exists
    if os.path.exists(PERSIST_DIRECTORY):
        print(f"\n   Removing old vector store at '{PERSIST_DIRECTORY}'")
        shutil.rmtree(PERSIST_DIRECTORY)

    print(f"\n  Creating vector store with {EMBEDDING_MODEL}")
    vector_store = Chroma.from_documents(
        documents=all_splits,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )

    print(f"Vector store saved to '{PERSIST_DIRECTORY}'.")


if __name__ == "__main__":
    ingest_documents()