import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

class LLMModule:
    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.95
        )
    
    def generate_response(self, prompt: str) -> str:
        """Generate response from the LLM"""
        response = self.pipeline(prompt)[0]["generated_text"]
        # Remove the prompt from the response
        return response[len(prompt):].strip()