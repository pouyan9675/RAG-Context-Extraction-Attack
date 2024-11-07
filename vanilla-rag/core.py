from typing import List, Dict
from database import VectorDB
from ui import ChatbotUI
from llm import LLMModule
from pathlib import Path
from typing import List
import logging



def setup_logging():
    """Configure logging settings"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

setup_logging()


class RAGSystem:
    def __init__(self, vector_store: VectorDB, llm: LLMModule):
        self.vector_store = vector_store
        self.llm = llm
    
    def add_knowledge(self, texts: List[str], metadata: List[Dict] = None):
        """Add documents to the knowledge base"""
        self.vector_store.add_documents(texts, metadata)
    
    def generate_response(self, query: str) -> str:
        """Generate response using RAG"""
        # Retrieve relevant documents
        relevant_docs = self.vector_store.search(query)
        
        # Sort by relevance score
        relevant_docs.sort(key=lambda x: x["score"])
        
        # Construct prompt with context
        context = "\n\n".join([
            f"[Document from {doc['metadata']['source']}]\n{doc['text']}"
            for doc in relevant_docs
        ])
        
        prompt = f"""Based on the following context, please answer the question.

        Context:
        {context}

        Question: {query}

        Answer:"""
        
        return self.llm.generate_response(prompt)
    



def index_documents(vector_store: VectorDB, input_path: str, recursive: bool = False) -> None:
    """Index documents from the specified path"""
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Path not found: {input_path}")
    
    files: List[Path] = []
    if input_path.is_file():
        files = [input_path]
    else:
        pattern = "**/*" if recursive else "*"
        files = [f for f in input_path.glob(pattern) if f.is_file()]
    
    logging.info(f"Found {len(files)} files to process")
    
    for file in files:
        try:
            logging.info(f"Processing: {file}")
            vector_store.add_document(file)
        except Exception as e:
            logging.error(f"Error processing {file}: {str(e)}")

def export_index(vector_store: VectorDB, output_path: str) -> None:
    """Export the vector store index"""
    output_path = Path(output_path)
    logging.info(f"Exporting index to: {output_path}")
    vector_store.export_index(output_path)

def import_index(vector_store: VectorDB, input_path: str) -> None:
    """Import a previously exported index"""
    input_path = Path(input_path)
    logging.info(f"Importing index from: {input_path}")
    vector_store.import_index(input_path)

