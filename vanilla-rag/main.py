#!/usr/bin/env python
import argparse
import logging
from ui import ChatbotUI
from core import RAGSystem


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
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="LLM model to use (default: meta-llama/Llama-3.2-1B-Instruct)"
    )
    
    parser.add_argument(
        "--embedding-model",
        default="dunzhang/stella_en_400M_v5",
        help="Embedding model to use (default: dunzhang/stella_en_400M_v5)"
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
    
    args = parser.parse_args()
    
    rag_system = RAGSystem(
        args.embedding_model,
        args.model,
    )
    ui = ChatbotUI(rag_system)
    
    # Execute command
    if args.command == "chat":
        run_chatbot(rag_system, ui)
    
    elif args.command == "index":
        rag_system.index_documents(
            args.input_path,
            recursive=args.recursive, 
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        rag_system.save_state()
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()