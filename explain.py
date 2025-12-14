from captum.attr import LayerIntegratedGradients
import torch

class XAIExplainer:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    # Helper to get embedding layer
    def _get_embedding_layer(self):
        if hasattr(self.model, "distilbert"):
            return self.model.distilbert.embeddings
        elif hasattr(self.model, "bert"):
            return self.model.bert.embeddings
        elif hasattr(self.model, "roberta"):
            return self.model.roberta.embeddings
        elif hasattr(self.model, "albert"):
            return self.model.albert.embeddings
        else:
            # Fallback: try to find a layer named 'embeddings' or 'wte'
            for name, module in self.model.named_modules():
                if "embeddings" in name and isinstance(module, torch.nn.Module):
                    return module
            raise ValueError(f"Could not automatically find embedding layer for {type(self.model)}")

    # Explain method
    def explain(self, text, target_label):
        self.model.eval()
        self.model.zero_grad()

        encoded = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        input_ids = encoded.input_ids.to(self.device)
        attention_mask = encoded.attention_mask.to(self.device)

        def forward_func(inputs, mask=None):
            return self.model(inputs, attention_mask=mask).logits

        embedding_layer = self._get_embedding_layer()
        lig = LayerIntegratedGradients(forward_func, embedding_layer)

        try:
            attributions = lig.attribute(
                inputs=input_ids,
                baselines=torch.zeros_like(input_ids),
                additional_forward_args=(attention_mask),
                target=target_label
            )
            
            attr_score = attributions.sum(dim=-1).squeeze(0)
            attr_score = attr_score / torch.norm(attr_score)
            
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
            scores = attr_score.detach().cpu().numpy()
            
            return list(zip(tokens, scores))
            
        except Exception as e:
            print(f"XAI Error: {e}")
            return []