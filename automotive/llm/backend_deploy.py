import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class LlamaBackend:
    def __init__(self, model_path, device="cuda"):
        """
        Args:
            model_path (str): Path to the model.
            device (str): Device to run on ('cuda' or 'cpu').
        """
        print(f"Loading Model: {model_path} on {device}...")
        self.device = device
        
        # Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Clean up memory before loading
        if device == "cuda":
            torch.cuda.empty_cache()

        # Load Model
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16
        )
        
        self.model.to(device)
        self.model.eval()
        print(f"Model Loaded Successfully. Memory: {self.model.get_memory_footprint() / 1024**3:.2f} GB")

    def predict(self, messages):
        """
        Runs inference and returns the decoded text string.
        """
        # 1. Tokenize
        inputs = self.tokenizer.apply_chat_template(
            messages, 
            return_tensors="pt", 
            add_generation_prompt=True
        ).to(self.device)

        if isinstance(inputs, dict) or hasattr(inputs, "input_ids"):
            input_ids = inputs["input_ids"]
            attention_mask = inputs.get("attention_mask", None)
        else:
            input_ids = inputs
            attention_mask = None

        # 2. Inference
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=5, 
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=False,
                temperature=None,
                top_p=None
            )
            
        # 3. Decode inside the backend
        generated_text = self.tokenizer.decode(
            output_ids[0][input_ids.shape[1]:], 
            skip_special_tokens=True
        ).strip()
            
        return generated_text
