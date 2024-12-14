import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


class Chatbot:
    """
        A class for out chatbot to generate responses and chat capabilities.
    """
    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.hidden_size = self.model.config.hidden_size
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.5,
            top_p=0.9,
        )
        self.message_history = [
            {"role": "system", "content": "You are a helpful assistant."},
        ]


    def generate_response(self, user_message: str) -> str:
        """Generate response from the LLM"""
        self.message_history.append(
            {"role": "user", "content": user_message},
        )
        intput_promt = self.tokenizer.apply_chat_template(self.message_history, 
                                                    add_generation_prompt=True)
        response = self.pipeline(intput_promt)[0]["generated_text"]
        response = response[len(intput_promt):].strip()
        self.message_history.append(
            {"role": "assistant", "content": response},
        )
        return response