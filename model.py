import torch
from transformers import AutoModelForSequenceClassification

# Generic model builder for any Transformer classifier.
def build_model(model_name, num_labels=2, device="cpu"):
    print(f"   Loading Architecture: {model_name}...")
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        output_attentions=False,
        output_hidden_states=False
    )
    
    model.to(device)
    return model