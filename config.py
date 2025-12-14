import torch

class Config:
    # --- Project Settings ---
    PROJECT_NAME = "NLP_Robustness_Benchmark"
    SEED = 42
    
    # Robust Device Logic
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        print(f"Device: CUDA (NVIDIA GPU) - {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
        print("Device: MPS (Apple Silicon)")
    else:
        DEVICE = torch.device("cpu")
        print("Device: CPU (Slow)")

    # --- Models to Compare ---
    TARGET_MODELS = {
        "DistilBERT": "distilbert-base-uncased",
        "BERT": "bert-base-uncased",
        "RoBERTa": "roberta-base"
    }
    
    # Default model name (required for initial data loading setup)
    MODEL_NAME = "distilbert-base-uncased"
    
    # The model used to GENERATE the attacks (usually kept constant for fair comparison)
    ATTACK_GEN_MODEL = "bert-base-uncased"
    
    OUTPUT_DIR = "./output"
    
    # --- Hyperparameters ---
    MAX_LEN = 128
    BATCH_SIZE = 16
    ACCUMULATION_STEPS = 2
    LEARNING_RATE = 2e-5
    EPOCHS = 1
    USE_FP16 = True
    
    # --- Attack Complexity ---
    # Set NUM_ATTACK_SAMPLES to None for full run, or 50-100 for debugging
    NUM_ATTACK_SAMPLES = 100
    MAX_REPLACEMENTS = 5
    K_NEIGHBORS = 8
    
    def __str__(self):
        return str(self.__class__.__dict__)