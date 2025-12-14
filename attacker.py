import torch
import random
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM

class ContextualAttacker:
    def __init__(self, target_model, target_tokenizer, config):
        self.target_model = target_model
        self.target_tokenizer = target_tokenizer
        self.config = config
        self.target_model.eval()
        
        # Load BERT-MLM for generating substitutes
        self.mlm_tokenizer = AutoTokenizer.from_pretrained(config.ATTACK_GEN_MODEL)
        self.mlm_model = AutoModelForMaskedLM.from_pretrained(config.ATTACK_GEN_MODEL).to(config.DEVICE)
        self.mlm_model.eval()

    # Predict the label of the target model
    def _predict_target(self, text):
        inputs = self.target_tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=self.config.MAX_LEN
        ).to(self.config.DEVICE)
        
        with torch.no_grad():
            outputs = self.target_model(**inputs)
            pred = torch.argmax(outputs.logits, dim=1).item()
        return pred

    # Generate substitutes using MLM
    def get_substitutes(self, text, index, k=10):
        tokens = self.mlm_tokenizer.tokenize(text)
        
        # Ensure index is within bounds
        if index >= 510: # 510 to account for [CLS] and [SEP]
            return []
            
        # Truncate tokens if they exceed BERT's max length
        if len(tokens) > 510:
            tokens = tokens[:510]
            
        if index >= len(tokens): return []
        
        tokens[index] = self.mlm_tokenizer.mask_token
        masked_text = self.mlm_tokenizer.convert_tokens_to_string(tokens)
        
        inputs = self.mlm_tokenizer(
            masked_text, 
            return_tensors="pt",
            truncation=True,
            max_length=512 # BERT-MLM supports up to 512
        ).to(self.config.DEVICE)
        
        with torch.no_grad():
            outputs = self.mlm_model(**inputs)
            
            # Find the mask index in the tokenized input
            mask_token_index = (inputs.input_ids == self.mlm_tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
            
            # If no mask token found, return empty list
            if len(mask_token_index) == 0: return []
            
            probs = outputs.logits[0, mask_token_index[0]]
            top_indices = torch.topk(probs, k).indices
            
            subs = []
            for tid in top_indices:
                word = self.mlm_tokenizer.decode([tid]).strip()
                if word.isalpha() and len(word) > 2:
                    subs.append(word)
        return subs

    # Perform the attack on a single text
    def attack(self, text, true_label):
        # Check initial prediction first
        original_pred = self._predict_target(text)
        if original_pred != true_label:
            return False, text, text

        words = text.split()
        
        # Shuffle indices to randomize attack order
        limit_len = min(len(words), self.config.MAX_LEN)
        indices = list(range(limit_len))
        random.shuffle(indices)
        
        current_text = text
        changes = 0

        for idx in indices:
            if changes >= self.config.MAX_REPLACEMENTS: break
            
            subs = self.get_substitutes(current_text, idx, k=self.config.K_NEIGHBORS)
            
            for syn in subs:
                temp_words = current_text.split()
                # Safety check
                if idx < len(temp_words):
                    temp_words[idx] = syn
                    temp_text = " ".join(temp_words)
                    
                    if self._predict_target(temp_text) != true_label:
                        return True, temp_text, text
        
        return False, text, text

    # Generate adversarial dataset
    def generate_adversarial_dataset(self, texts, labels, desc="Attacking"):
        adv_data = []
        limit = len(texts) if self.config.NUM_ATTACK_SAMPLES is None else min(len(texts), self.config.NUM_ATTACK_SAMPLES)
        
        success_count = 0
        for i in tqdm(range(limit), desc=desc):
            success, adv_text, orig_text = self.attack(texts[i], labels[i])
            if success:
                success_count += 1
                adv_data.append({
                    "original_text": orig_text,
                    "adv_text": adv_text,
                    "label": labels[i]
                })
        
        asr = (success_count / limit) * 100 if limit > 0 else 0
        return adv_data, asr