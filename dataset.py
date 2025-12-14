import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

# Function to load and preprocess the IMDb dataset
def get_dataloaders(cfg, debug=False):
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME)
    
    print("Loading IMDb dataset...")
    dataset = load_dataset("imdb")
    
    if debug:
        print("DEBUG MODE: Using tiny dataset subset.")
        # We use a small subset for debugging
        train_data = dataset['train'].shuffle(seed=cfg.SEED).select(range(cfg.NUM_ATTACK_SAMPLES))
        test_data = dataset['test'].shuffle(seed=cfg.SEED).select(range(cfg.NUM_ATTACK_SAMPLES))
    else:
        # Full training set; subset of test set for faster evaluation in project
        train_data = dataset['train']
        test_data = dataset['test'].shuffle(seed=cfg.SEED)
        
    print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")

    # Define Tokenization Logic
    def tokenize_function(examples):
        return tokenizer(
            examples['text'], 
            truncation=True, 
            padding="max_length", 
            max_length=cfg.MAX_LEN
        )

    # Apply Tokenization using .map() for efficiency
    print("Tokenizing datasets...")
    tokenized_train = train_data.map(tokenize_function, batched=True)
    tokenized_test = test_data.map(tokenize_function, batched=True)

    # Rename 'label' to 'labels' for compatibility with Transformers
    tokenized_train = tokenized_train.rename_column("label", "labels")
    tokenized_test = tokenized_test.rename_column("label", "labels")

    # Set Format to Torch Tensors
    tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    tokenized_test.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # Create DataLoaders
    train_loader = DataLoader(tokenized_train, batch_size=cfg.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(tokenized_test, batch_size=cfg.BATCH_SIZE, shuffle=False)
    
    # Return loaders + raw text (needed for the Attacker)
    return train_loader, test_loader, tokenizer, train_data['text'], train_data['label']