import gradio as gr
from core import RAGSystem

class ChatbotUI:
    """
        Designs a simple gradio User Interface in order to use Chatbot with 
        Retrieval Augmented Generation
    """
    def __init__(self, rag_system: RAGSystem):
        self.rag_system = rag_system
        
    def add_documents(self, text: str) -> str:
        """Add documents through the UI"""
        documents = [doc.strip() for doc in text.split("\n\n") if doc.strip()]
        self.rag_system.add_knowledge(documents)
        return f"Added {len(documents)} documents to the knowledge base."
    
    def query(self, question: str) -> str:
        """Query the RAG system through the UI"""
        return self.rag_system.generate_response(question)
    
    def create_interface(self):
        """Create the Gradio interface"""
        with gr.Blocks() as interface:
            gr.Markdown("# RAG System Demo")
            
            with gr.Tab("Add Knowledge"):
                text_input = gr.Textbox(
                    lines=10,
                    label="Enter documents (separate with blank lines)"
                )
                add_btn = gr.Button("Add Documents")
                add_output = gr.Textbox(label="Status")
                add_btn.click(
                    fn=self.add_documents,
                    inputs=[text_input],
                    outputs=[add_output]
                )
            
            with gr.Tab("Query"):
                question_input = gr.Textbox(label="Your Question")
                query_btn = gr.Button("Ask")
                answer_output = gr.Textbox(label="Answer")
                query_btn.click(
                    fn=self.query,
                    inputs=[question_input],
                    outputs=[answer_output]
                )
        
        return interface