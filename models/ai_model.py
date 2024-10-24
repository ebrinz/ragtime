import torch
from transformers import AutoModel, AutoTokenizer

class BaseModel:
    
    def __init__(self, model_name: str, device: str = None):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = device or torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = None

    def to_device(self):
        if self.model:
            self.model.to(self.device)


class GenerationModel(BaseModel):

    def load_model(self):
        from transformers import AutoModelForCausalLM
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16, 
            device_map={"": self.device}, 
            trust_remote_code=True
            )
        self.model.to(self.device)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.eos_token})

    def generate_response(self, input_text: str, **generation_kwargs):
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                pad_token_id=self.tokenizer.pad_token_id,
                **generation_kwargs
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


class EmbeddingsModel(BaseModel):

    def load_model(self):
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)

    def get_batch_embeddings(self, batch_texts: list, remove_token_type_ids=False, mean_pooling=True):
        inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        if remove_token_type_ids and 'token_type_ids' in inputs:
            del inputs['token_type_ids']
        with torch.no_grad():
            outputs = self.model(**inputs)
        if mean_pooling:
            return outputs.last_hidden_state.mean(dim=1)
        return outputs.last_hidden_state
