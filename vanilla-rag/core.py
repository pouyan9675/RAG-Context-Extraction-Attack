import logging
from tqdm import tqdm
from llm import Chatbot
from typing import List
from pathlib import Path
from database import VectorDB
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer


def setup_logging():
    """Configure logging settings"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

setup_logging()


class RAGSystem:
    def __init__(
        self,
        embedding_model_id: str,
        chatbot_model_id: str
    ):
        self.chat_model = Chatbot(chatbot_model_id)
        self.embedding_model = SentenceTransformer(
            embedding_model_id,
            trust_remote_code=True, 
            device="cuda"
        )
        self.vector_db = VectorDB(
            dimension=self.embedding_model[-1].out_features,
        )

    
    def add_knowledge(self, documents: dict[str, str], metadata: List[Dict] = None):
        """Add documents to the knowledge base"""
        for id_, text in documents.items():
            # TODO: correct this one
            self.vector_db.add(id_, self.embedding_model(text))


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

    def _read_file(self, file_path: Path) -> str:
        """Read content from a file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _create_chunks(
        self,
        text: str,
        chunk_size: int,
        chunk_overlap: int
    ) -> List[str]:
        """Split text into overlapping chunks"""
        if not text:
            return []
        
        if chunk_size is None or chunk_size <= 0:
            return [text]
            
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            # Calculate end position for current chunk
            end = start + chunk_size
            
            # If this is not the first chunk, start earlier to create overlap
            if start > 0:
                start = max(0, start - chunk_overlap)
            
            # Extract chunk
            chunk = text[start:min(end, text_length)]
            
            # Add chunk if it's not empty
            if chunk.strip():
                chunks.append(chunk)
            
            # Move to next chunk starting position
            start = end

        return chunks

    def _process_chunk(
        self,
        chunk: str,
        file_path: Path,
        chunk_index: int
    ) -> None:
        """Process a single chunk of text"""
        try:
            # Generate embedding for the chunk
            embedding = self.embedding_model.encode(chunk, show_progress_bar = False)
            
            # Create a unique ID for the chunk
            chunk_id = f"{file_path}_{chunk_index}"
            
            # Add metadata about the chunk
            metadata = {
                "source_file": str(file_path),
                "chunk_index": chunk_index,
                "chunk_text": chunk[:100] + "..." if len(chunk) > 100 else chunk
            }
            # Add to vector database
            self.vector_db.add(
                id=chunk_id,
                vector=embedding,
                metadata=metadata
            )
            
        except Exception as e:
            logging.error(f"Error processing chunk {chunk_index} from {file_path}: {str(e)}")


    def index_documents(
        self,
        input_path: str,
        recursive: bool = False,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> None:
        """
        Index documents from the specified path with chunking support
        
        Args:
            input_path: Path to file or directory to index
            recursive: Whether to recursively process directories
            chunk_size: Size of each text chunk (in characters). If None, no chunking is performed
            chunk_overlap: Number of characters to overlap between chunks. If None, defaults to 10% of chunk_size
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Path not found: {input_path}")
        
        # Set default overlap if chunking is enabled but overlap not specified
        if chunk_size and chunk_overlap is None:
            chunk_overlap = max(1, int(chunk_size * 0.1))  # 10% overlap by default
        
        files: List[Path] = []
        if input_path.is_file():
            files = [input_path]
        else:
            pattern = "**/*" if recursive else "*"
            files = [f for f in input_path.glob(pattern) if f.is_file()]
        
        logging.info(f"Found {len(files)} files to process")
        
        for file in tqdm(files, desc="Processing files"):
            try:
                # Read file content
                text = self._read_file(file)
                
                if chunk_size:
                    # Process file in chunks
                    chunks = self._create_chunks(text, chunk_size, chunk_overlap)
                    # logging.info(f"Created {len(chunks)} chunks from {file}")
                    
                    for i, chunk in enumerate(chunks):
                        self._process_chunk(chunk, file, i)
                else:
                    # Process entire file as one chunk
                    self._process_chunk(text, file, 0)
                    
            except Exception as e:
                logging.error(f"Error indexing {file}: {str(e)}")


    def save_state(self,):
        """
            Save the current state of the RAG system such as vector DB and etc.
        """
        self.vector_db.store()

