# Khmer ⇄ English Romanization

A bidirectional transliteration system between Khmer and English using deep learning models (LSTM and Transformer architectures). This project provides both sequence-to-sequence LSTM models and Transformer models for transliterating between Khmer script and romanized English.

## Table of Contents

- [Overview](#overview)
- [Software Dependencies](#software-dependencies)
- [Hardware Requirements](#hardware-requirements)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Training Models](#training-models)
- [Running Evaluation](#running-evaluation)
- [Using the Demo Application](#using-the-demo-application)
- [Project Structure](#project-structure)
- [Model Architectures](#model-architectures)

## Overview

This project implements four deep learning models:

1. **LSTM English → Khmer** (`khmer_transliterator.keras`)
2. **LSTM Khmer → English** (`english_romanizer.keras`)
3. **Transformer English → Khmer** (`transformer_eng2khm.keras`)
4. **Transformer Khmer → English** (`transformer_romanizer.keras`)

All models use character-level tokenization and seq2seq architecture for transliteration.

## Software Dependencies

### requirements.txt

```
tensorflow>=2.13.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
gradio>=3.40.0
editdistance>=0.6.2
jupyter>=1.0.0
notebook>=7.0.0
```

### Python Version

- Python 3.9 or higher

### System Dependencies

- Linux, macOS, or Windows with WSL2
- CUDA Toolkit 11.8+ (for GPU acceleration, optional but recommended)
- cuDNN 8.6+ (for GPU acceleration, optional but recommended)

## Hardware Requirements

### Minimum Requirements

- **CPU**: 4 cores (Intel i5 or AMD Ryzen 5 equivalent)
- **RAM**: 8 GB
- **GPU**: optional (but recommended)
- **Storage**: 2 GB free disk space
- **Training Time**: 2-3 hours per model on CPU

## Installation

### Step 1: Clone or Navigate to the Project Directory

```bash
git clone 'https://github.com/LazyGreed/khm_romanization'
cd khm_romanization
```

### Step 2: Create a Python Virtual Environment

```bash
python3 -m venv .venv
```

or using `uv`

```bash
uv venv .venv
```

### Step 3: Activate the Virtual Environment

**On Linux/macOS:**
```bash
source .venv/bin/activate
```

**On Windows:**
```bash
.venv\Scripts\activate
```

### Step 4: Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

or using `uv`

```bash
uv pip install -r requirements.txt
```

## Data Preparation

The project uses the dataset at `data/raw/eng_khm_data.csv` which contains English-Khmer transliteration pairs.

### Data Format

The CSV file contains two columns:
- `khm`: Khmer script text
- `eng`: Romanized English text

### Verify Data

```bash
head -n 5 data/raw/eng_khm_data.csv
```

The dataset should contain approximately 28,577 pairs of Khmer-English transliterations.

### Data Statistics

```bash
wc -l data/raw/eng_khm_data.csv
```

## Training Models

All models can be trained using the Jupyter notebooks in the `notebooks/` directory. Training creates model files in `models/` and preprocessing assets in `data/processed/`.

### Step 1: Start Jupyter Notebook Server

```bash
jupyter notebook
```

This will open a browser window with the Jupyter interface.

### Step 2: Train LSTM Models

#### 2.1: Train English → Khmer LSTM Model

1. Open `notebooks/01_eng2khm_seq2seq.ipynb`
2. Run all cells sequentially (Cell → Run All)
3. **Output:**
   - Model: `models/khmer_transliterator.keras`
   - Assets: `data/processed/khmer_transliteration_assets.pkl`
4. **Training time:** ~20-30 minutes on GPU, ~2-3 hours on CPU
5. **Epochs:** 50

#### 2.2: Train Khmer → English LSTM Model

1. Open `notebooks/02_khm2eng_seq2seq.ipynb`
2. Run all cells sequentially (Cell → Run All)
3. **Output:**
   - Model: `models/english_romanizer.keras`
   - Assets: `data/processed/english_romanization_assets.pkl`
4. **Training time:** ~20-30 minutes on GPU, ~2-3 hours on CPU
5. **Epochs:** 50

### Step 3: Train Transformer Models

#### 3.1: Train English → Khmer Transformer Model

1. Open `notebooks/05_eng2khm_transformer.ipynb`
2. Run all cells sequentially (Cell → Run All)
3. **Output:**
   - Model: `models/transformer_eng2khm.keras`
   - Assets: `data/processed/transformer_eng2khm_assets.pkl`
4. **Training time:** ~15-25 minutes on GPU, ~1.5-2.5 hours on CPU
5. **Epochs:** 50

#### 3.2: Train Khmer → English Transformer Model

1. Open `notebooks/04_khm2eng_transformer.ipynb`
2. Run all cells sequentially (Cell → Run All)
3. **Output:**
   - Model: `models/transformer_romanizer.keras`
   - Assets: `data/processed/transformer_romanization_assets.pkl`
4. **Training time:** ~15-25 minutes on GPU, ~1.5-2.5 hours on CPU
5. **Epochs:** 50

### Training Configuration

All models use the following hyperparameters:

**LSTM Models:**
- Embedding dimension: 32
- LSTM units: 64
- Batch size: 16
- Epochs: 50
- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy

**Transformer Models:**
- Embedding dimension: 128
- Latent dimension: 256
- Number of attention heads: 4
- Batch size: 64
- Epochs: 50
- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy

## Running Evaluation

Evaluation scripts compute various metrics including Word Accuracy, Character Accuracy, F1 Score, and Character Error Rate (CER).

### Step 1: Evaluate LSTM Models

1. Open `notebooks/03_evaluation.ipynb`
2. Run all cells sequentially (Cell → Run All)
3. **Output:**
   - `results/eng2khm/eng2khm_evaluation.csv`
   - `results/khm2eng/khm2eng_evaluation.csv`
   - Visualization plots in respective directories
4. **Evaluation time:** ~5-10 minutes

The notebook evaluates both:
- English → Khmer LSTM model
- Khmer → English LSTM model

### Step 2: Evaluate Transformer Models

1. Open `notebooks/06_transformer_evaluation.ipynb`
2. Run all cells sequentially (Cell → Run All)
3. **Output:**
   - `results/eng2khm/transformer_eng2khm_evaluation.csv`
   - `results/khm2eng/transformer_khm2eng_evaluation.csv`
   - `results/transformer_models_comparison.png`
   - Visualization plots in respective directories
4. **Evaluation time:** ~5-10 minutes

The notebook evaluates both:
- English → Khmer Transformer model
- Khmer → English Transformer model

### Evaluation Metrics

The evaluation computes:
- **Word Accuracy**: Percentage of exact matches
- **Character Accuracy**: 1 - (edit_distance / max_length)
- **F1 Score**: Harmonic mean of precision and recall at character level
- **CER (Character Error Rate)**: edit_distance / target_length

### View Evaluation Results

```bash
# View LSTM results
cat results/eng2khm/eng2khm_evaluation.csv | head -n 20
cat results/khm2eng/khm2eng_evaluation.csv | head -n 20

# View Transformer results
cat results/eng2khm/transformer_eng2khm_evaluation.csv | head -n 20
cat results/khm2eng/transformer_khm2eng_evaluation.csv | head -n 20
```

## Using the Demo Application

The demo application provides an interactive web interface for transliteration using both LSTM and Transformer models.

### Step 1: Ensure All Models Are Trained

Verify that all required models exist:

```bash
ls -lh models/
```

You should see:
- `khmer_transliterator.keras` (LSTM Eng→Khm)
- `english_romanizer.keras` (LSTM Khm→Eng)
- `transformer_eng2khm.keras` (Transformer Eng→Khm)
- `transformer_romanizer.keras` (Transformer Khm→Eng)

### Step 2: Launch the Demo Application

```bash
python demo.py
```

or using `uv`

```bash
uv run demo.py
```

### Step 3: Access the Web Interface

The application will start and display output like:

```
Loading LSTM models...
LSTM models loaded successfully!
Running on local URL:  http://127.0.0.1:7860

To create a public link, set `share=True` in `launch()`.
```

Open your browser and navigate to: `http://127.0.0.1:7860`

### Step 4: Use the Application

1. **Select Model Type:** Choose between "lstm" or "transformer" using the radio buttons
2. **Enter Text:** Type English or Khmer text in the input box
   - For English input: Use lowercase letters (e.g., "hello", "mean", "kdar")
   - For Khmer input: Use Khmer script (e.g., "ហេឡូ", "មាន", "ក្ដា")
3. **Click "Romanize"** or press Enter
4. **View Results:** 
   - Output text appears in the output box
   - Detection info shows which direction was used

### Example Inputs

**English → Khmer:**
- `hello` → `ហេឡូ`
- `mean` → `មាន`
- `kdar` → `ក្ដា`

**Khmer → English:**
- `ហេឡូ` → `hello`
- `មាន` → `mean`
- `ក្ដា` → `kdar`

### Stop the Demo Application

Press `Ctrl+C` in the terminal to stop the server.

## Project Structure

```
khmer_romanization/
├── README.md                          # This file
├── demo.py                            # Gradio web application
├── requirements.txt                   # Python dependencies
├── data/
│   ├── raw/
│   │   ├── eng_khm_data.csv          # Training dataset (28,577 pairs)
│   │   └── data_kh.csv               # Additional data (if any)
│   └── processed/
│       ├── khmer_transliteration_assets.pkl      # LSTM Eng→Khm assets
│       ├── english_romanization_assets.pkl       # LSTM Khm→Eng assets
│       ├── transformer_eng2khm_assets.pkl        # Transformer Eng→Khm assets
│       └── transformer_romanization_assets.pkl   # Transformer Khm→Eng assets
├── models/
│   ├── khmer_transliterator.keras              # LSTM Eng→Khm model
│   ├── english_romanizer.keras                 # LSTM Khm→Eng model
│   ├── transformer_eng2khm.keras               # Transformer Eng→Khm model
│   └── transformer_romanizer.keras             # Transformer Khm→Eng model
├── notebooks/
│   ├── 01_eng2khm_seq2seq.ipynb               # Train LSTM Eng→Khm
│   ├── 02_khm2eng_seq2seq.ipynb               # Train LSTM Khm→Eng
│   ├── 03_evaluation.ipynb                     # Evaluate LSTM models
│   ├── 04_khm2eng_transformer.ipynb           # Train Transformer Khm→Eng
│   ├── 05_eng2khm_transformer.ipynb           # Train Transformer Eng→Khm
│   └── 06_transformer_evaluation.ipynb         # Evaluate Transformer models
├── results/
│   ├── eng2khm/
│   │   ├── eng2khm_evaluation.csv             # LSTM Eng→Khm results
│   │   ├── transformer_eng2khm_evaluation.csv # Transformer Eng→Khm results
│   │   └── *.png                              # Visualizations
│   └── khm2eng/
│       ├── khm2eng_evaluation.csv             # LSTM Khm→Eng results
│       ├── transformer_khm2eng_evaluation.csv # Transformer Khm→Eng results
│       └── *.png                              # Visualizations
└── documents/
    └── lstm_model.md                          # Model documentation
```

## Model Architectures

### LSTM Seq2Seq Architecture

**Encoder:**
- Input: Character-level tokenized sequences
- Embedding layer (dimension: 32)
- LSTM layer (64 units)
- Output: Hidden states (h, c)

**Decoder:**
- Input: Start token + target sequence
- Embedding layer (dimension: 32)
- LSTM layer (64 units, initialized with encoder states)
- Dense layer with softmax activation
- Output: Character probabilities

**Training:**
- Teacher forcing during training
- Greedy decoding during inference

### Transformer Architecture

**Encoder:**
- Input: Character-level tokenized sequences
- Embedding layer (dimension: 128)
- Dense layer (dimension: 256, ReLU activation)

**Decoder:**
- Input: Start token + target sequence
- Embedding layer (dimension: 128)
- Dense layer (dimension: 256, ReLU activation)
- Multi-head attention (4 heads, key dimension: 256)
- Layer normalization
- Dropout (0.2)
- Dense output layer with softmax activation

**Training:**
- Teacher forcing during training
- Greedy decoding during inference

## Complete Workflow Summary

Here's the complete step-by-step workflow from installation to evaluation:

```bash
# 1. Setup environment
git clone https://github.com/LazyGreed/khm_romanization
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Train models (via Jupyter notebooks)
jupyter notebook
# Then open and run:
#   - notebooks/01_eng2khm_seq2seq.ipynb
#   - notebooks/02_khm2eng_seq2seq.ipynb
#   - notebooks/04_khm2eng_transformer.ipynb
#   - notebooks/05_eng2khm_transformer.ipynb

# 4. Run evaluation (via Jupyter notebooks)
# Open and run:
#   - notebooks/03_evaluation.ipynb
#   - notebooks/06_transformer_evaluation.ipynb

# 5. View results
ls -lh models/
ls -lh results/eng2khm/
ls -lh results/khm2eng/

# 6. Launch demo application
python demo.py
# Then open browser to http://127.0.0.1:7860
```

## Troubleshooting

### Out of Memory Errors

If you encounter OOM errors during training:

1. Reduce batch size in the notebooks:
   - LSTM: Change `BATCH_SIZE = 16` to `BATCH_SIZE = 8`
   - Transformer: Change `BATCH_SIZE = 64` to `BATCH_SIZE = 32`

2. Close other GPU-intensive applications

### Slow Training on CPU

Training on CPU is significantly slower. Consider:

1. Reducing the number of epochs (e.g., from 50 to 20)
2. Using a smaller subset of the data for testing

### Demo Application Not Loading Models

Ensure all four model files and their corresponding asset files exist:

```bash
ls -lh models/*.keras
ls -lh data/processed/*.pkl
```

If any are missing, retrain the corresponding model using its notebook.

## References

- https://github.com/Chhunneng/khmer-text-transliteration
