import re
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

class AIModel:
    def __init__(self, model_name, device='mps', is_generation_model=False):
        self.model_name = model_name
        self.device = torch.device(device)
        self.is_generation_model = is_generation_model
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.is_generation_model:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
        else:
            self.model = AutoModel.from_pretrained(self.model_name).to(self.device)

        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.eos_token})

    def tokenize(self, texts, max_length=512, remove_token_type_ids=False):
        """Tokenizes the input text"""
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(self.device)
        if remove_token_type_ids and 'token_type_ids' in inputs:
            del inputs['token_type_ids']
        return inputs

    def generate(self, texts, max_length=500, **kwargs):
        """Generates text using a text generation model"""
        if not self.is_generation_model:
            raise ValueError("Model is not set up for text generation.")
        inputs = self.tokenize(texts, max_length=max_length)
        with torch.no_grad():
            outputs = self.model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=max_length,
                **kwargs
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def get_embeddings(self, texts, mean_pooling=True, **kwargs):
        if self.is_generation_model:
            raise ValueError("Model is not set up for embeddings.")
        inputs = self.tokenize(texts, **kwargs)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1) if mean_pooling else outputs.last_hidden_state

    def clean_plot(self, plot):
        plot = re.sub(r'\[.*?\]', '', plot)
        plot = re.sub(r'\s+', ' ', plot)
        plot = plot.strip()
        return plot.lower()
