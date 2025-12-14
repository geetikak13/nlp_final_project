import torch
from torch.optim import AdamW
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast

# Training function with Mixed Precision and Gradient Accumulation
def train_model(model, train_loader, config):
    model.train()
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)
    
    # Initialize Scaler for FP16 (Mixed Precision)
    scaler = GradScaler(enabled=config.USE_FP16)

    for epoch in range(config.EPOCHS):
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS}", leave=False)
        for i, batch in enumerate(loop):
            input_ids = batch['input_ids'].to(config.DEVICE)
            attention_mask = batch['attention_mask'].to(config.DEVICE)
            labels = batch['labels'].to(config.DEVICE)

            # Mixed Precision Context
            with autocast(enabled=config.USE_FP16):
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / config.ACCUMULATION_STEPS

            # Backward pass with scaler
            scaler.scale(loss).backward()

            # Gradient Accumulation Step
            if (i + 1) % config.ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
            loop.set_postfix(loss=loss.item() * config.ACCUMULATION_STEPS)
            
    return model

# Evaluation function
def evaluate_model(model, loader, config):
    model.eval()
    preds, true_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(config.DEVICE)
            attention_mask = batch['attention_mask'].to(config.DEVICE)
            labels = batch['labels'].to(config.DEVICE)

            with autocast(enabled=config.USE_FP16):
                outputs = model(input_ids, attention_mask=attention_mask)
            
            preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    return accuracy_score(true_labels, preds)