import torch
import numpy as np
import random
import nltk
import os

# Utility functions
def set_seed(seed_val):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    torch.backends.cudnn.deterministic = True

# Download necessary NLTK data files
def download_nltk_data():
    resources = ['wordnet', 'omw-1.4', 'averaged_perceptron_tagger']
    for res in resources:
        try:
            nltk.data.find(f'corpora/{res}')
        except LookupError:
            print(f"Downloading NLTK resource: {res}")
            nltk.download(res, quiet=True)

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)