from sentence_transformers import SentenceTransformer, util
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

class AdvancedMetrics:
    def __init__(self, device):
        print("Loading Advanced Metrics Models (SentenceTransformer & GPT-2)...")
        # Lightweight model for cosine similarity
        self.sim_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        
        # GPT-2 for Perplexity (Fluency)
        self.ppl_model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
        self.ppl_tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        self.device = device

    # Cosine Similarity
    def calculate_similarity(self, original_text, adv_text):
        emb1 = self.sim_model.encode(original_text, convert_to_tensor=True)
        emb2 = self.sim_model.encode(adv_text, convert_to_tensor=True)
        return util.pytorch_cos_sim(emb1, emb2).item()
    
    # Perplexity
    def calculate_perplexity(self, text):
        encodings = self.ppl_tokenizer(text, return_tensors='pt').to(self.device)
        max_len = self.ppl_model.config.n_positions
        if encodings.input_ids.size(1) > max_len:
            return float("inf") # Skip if too long

        with torch.no_grad():
            outputs = self.ppl_model(**encodings, labels=encodings.input_ids)
            loss = outputs.loss
        return torch.exp(loss).item()