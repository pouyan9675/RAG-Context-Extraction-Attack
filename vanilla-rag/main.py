import argparse
import logging
from ui import ChatbotUI
from database import VectorDB
from llm import LLMModule
from core import (
    RAGSystem,
    index_documents,
    import_index,
    export_index,
)


def setup_logging():
    """Configure logging settings"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

setup_logging()


def run_chatbot(rag_system: RAGSystem, ui: ChatbotUI) -> None:
    """Launch the RAG chatbot interface"""
    logging.info("Starting RAG chatbot interface")
    interface = ui.create_interface()
    interface.launch()


def main():
    parser = argparse.ArgumentParser(
        description="RAG System CLI",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        "--model",
        default="gpt-3.5-turbo",
        help="LLM model to use (default: gpt-3.5-turbo)"
    )
    
    parser.add_argument(
        "--embedding-model",
        default="text-embedding-ada-002",
        help="Embedding model to use (default: text-embedding-ada-002)"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Start the RAG chatbot")
    chat_parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the interface on"
    )
    
    # Index command
    index_parser = subparsers.add_parser("index", help="Index documents")
    index_parser.add_argument(
        "input_path",
        help="Path to file or directory to index"
    )
    index_parser.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        help="Recursively process directories"
    )
    index_parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Document chunk size (default: 1000)"
    )
    index_parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Overlap between chunks (default: 200)"
    )
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export vector store index")
    export_parser.add_argument(
        "output_path",
        help="Path to export the index"
    )
    
    # Import command
    import_parser = subparsers.add_parser("import", help="Import vector store index")
    import_parser.add_argument(
        "input_path",
        help="Path to the index file to import"
    )
    
    args = parser.parse_args()
    
    
    # Initialize components
    vector_store = VectorDB(
        embedding_model=args.embedding_model
    )
    
    llm = LLMModule(
        model=args.model
    )
    
    rag_system = RAGSystem(vector_store, llm)
    ui = ChatbotUI(rag_system)
    
    # Execute command
    if args.command == "chat":
        run_chatbot(rag_system, ui)
    
    elif args.command == "index":
        index_documents(
            vector_store,
            args.input_path,
            recursive=args.recursive
        )
    
    elif args.command == "export":
        export_index(vector_store, args.output_path)
    
    elif args.command == "import":
        import_index(vector_store, args.input_path)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()