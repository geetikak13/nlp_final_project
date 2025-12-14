# Automated Robustness Evaluation of Transformer Models

## Project Overview
This project benchmarks the robustness of NLP models (DistilBERT, BERT, RoBERTa) against adversarial text attacks and explores defense mechanisms to improve resilience. It implements a **Contextual Synonym Substitution Attack** on sentiment classifiers trained on the IMDb dataset.

Key components include:
* **Attack:** A black-box attack that greedily substitutes words with synonyms (via BERT-MLM) to flip the model's prediction while preserving semantic meaning.
* **Defense:** Adversarial Training (Data Augmentation) to harden the models.
* **Evaluation:** Metrics include Attack Success Rate (ASR), Semantic Similarity (Sentence-BERT), and Fluency (GPT-2 Perplexity).
* **Explainability (XAI):** Integration with **Captum** (Integrated Gradients) to visualize how adversarial perturbations shift model attention.
* **Interface:** A graphical demo using **Gradio** for real-time testing.

## Project Structure
```text
nlp_final_project/
├── output/                   # Saved models, logs, and benchmark CSVs
├── attacker.py               # Contextual Adversarial Attack Logic
├── config.py                 # Configuration (Hyperparameters, Paths)
├── dataset.py                # Data loading & Tokenization
├── explain.py                # XAI (Captum Integrated Gradients)
├── metrics.py                # Semantic Sim & Perplexity Calculators
├── model.py                  # Model Wrapper Factory
├── project_walkthrough.ipynb # MAIN LAUNCHER: Interactive Notebook including Gradio Web Interface (Bonus Demo)
├── requirements.txt          # Dependencies
├── trainer.py                # Mixed-Precision Training Loop
└── utils.py                  # Helper functions
```

## Setup & Installation

1.  **Clone the repository :**
    ```bash
    git clone https://github.com/geetikak13/cnn_performance_comparison
    ```

2. **Environment Setup**
Create a virtual environment to isolate dependencies:
```Bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate
```

3.  **Install dependencies:**
```Bash
pip install -r requirements.txt
# Or use: python3 -m pip install -r requirements.txt
```
**Note: This installs the following requirements**
* torch
* transformers
* datasets
* nltk
* scikit-learn
* numpy
* tqdm
* sentence-transformers
* captum
* matplotlib
* NVIDIA GPU with CUDA support and compatible drivers.
* Apple Silicon Mac with macOS 12.3+.


## 🚀 Setup & Installation
1. Prerequisites
* Python 3.8+
* CUDA-enabled GPU (Highly recommended for training)

2. Environment Setup
Create a virtual environment to isolate dependencies:
```Bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate
```


## Usage
1. **Interactive Walkthrough (Main Launcher)**
This is the primary way to run the project. It guides you through the entire pipeline step-by-step, including training, attacking, and visualizing attention maps.
  1. Start Jupyter:
```Bash
  jupyter notebook
```

  2. Open project_walkthrough.ipynb.
  3. Run the cells sequentially. 
    * You will see:Baseline model training.
    * Live generation of adversarial examples.
    * Heatmaps showing word importance shifting after an attack.
    * Adversarial training and re-evaluation.
    * The final cell has an built-in **Web Interface (Demo)** to interact with the model in a user-friendly GUI.
      * Analyze Sentiment: Type a review to see the confidence score.
      * Launch Attack: Click to generate an adversarial example that fools the model.
      * Explain: Visualize which words contributed most to the prediction using Integrated Gradients.
