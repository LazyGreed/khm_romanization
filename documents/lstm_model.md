# Khmer-English Transliteration Models

This document describes the architecture, training process, and evaluation methods for two bidirectional sequence-to-sequence models for Khmer-English transliteration.

## Overview

Two complementary models have been developed:

1. **English to Khmer Transliterator** (`khmer_transliterator.keras`) - Converts romanized English text to Khmer script
2. **Khmer to English Romanizer** (`english_romanizer.keras`) - Converts Khmer script to romanized English text

Both models use identical architectures but are trained on reversed data pairs to enable bidirectional transliteration.

## Model Architecture

### Architecture Type: Sequence-to-Sequence (Seq2Seq) with LSTM

Both models implement a classic encoder-decoder architecture with the following components:

#### Encoder
- **Input Layer**: Variable-length sequence of character indices
- **Embedding Layer**: 
  - Converts character indices to dense vectors
  - Embedding dimension: 32
  - Vocabulary size: Language-specific (based on unique characters in training data)
- **LSTM Layer**:
  - Units: 64
  - Returns final hidden state (h) and cell state (c)
  - These states capture the semantic meaning of the input sequence

#### Decoder
- **Input Layer**: Variable-length sequence of target character indices
- **Embedding Layer**:
  - Embedding dimension: 32
  - Target vocabulary size: Language-specific
- **LSTM Layer**:
  - Units: 64
  - Initialized with encoder's final states (h, c)
  - Returns full sequence of outputs
- **Dense Layer**:
  - Units: Target vocabulary size
  - Activation: Softmax
  - Outputs probability distribution over target characters

### Model Configuration

```python
EMBED_DIM = 32       # Embedding dimension for character vectors
LSTM_UNITS = 64      # Number of LSTM units in encoder/decoder
BATCH_SIZE = 16      # Training batch size
EPOCHS = 50          # Training epochs
BEAM_WIDTH = 3       # Beam width parameter (defined but not implemented)
```

### Input/Output Format

#### English to Khmer Model
- **Input**: Romanized English text (lowercase, a-z only)
- **Output**: Khmer Unicode characters (U+1780 to U+17FF)

#### Khmer to English Model
- **Input**: Khmer Unicode characters (U+1780 to U+17FF)
- **Output**: Romanized English text (lowercase, a-z only)

## Data Preprocessing

### Data Loading
- **Source**: `data/raw/eng_khm_data.csv`
- **Format**: CSV file with columns `eng` (English) and `khm` (Khmer)

### Text Normalization

#### English Text
```python
normalized_eng = re.sub(r"[^a-z]", "", row['eng'].lower())
```
- Convert to lowercase
- Remove all non-alphabetic characters (keep only a-z)

#### Khmer Text
```python
normalized_khm = re.sub(r"[^\u1780-\u17FF]", "", row['khm'])
normalized_khm = unicodedata.normalize('NFC', normalized_khm)
```
- Keep only Khmer Unicode characters (U+1780-U+17FF)
- Apply Unicode NFC normalization for consistent character representation

### Tokenization

Both models use character-level tokenization:

- **Tokenizer Type**: Keras `Tokenizer` with `char_level=True`
- **Special Tokens**: 
  - `<unk>`: Out-of-vocabulary token
  - `\t`: Start-of-sequence token (for decoder input)
  - `\n`: End-of-sequence token (for decoder target)
- **Filters**: None (`filters=''`) to preserve all characters after normalization

### Sequence Preparation

For each training pair:

1. **Encoder Input**: 
   - Tokenized source text
   - Padded to `max_source_len` with post-padding

2. **Decoder Input**: 
   - `[\t] + tokenized_target_text`
   - Padded to `max_target_len + 1` with post-padding

3. **Decoder Target**: 
   - `tokenized_target_text + [\n]`
   - Padded to `max_target_len + 1` with post-padding

This teacher forcing approach provides the correct previous character to the decoder during training.

## Training Details

### Optimization
- **Optimizer**: Adam (default parameters)
- **Loss Function**: Sparse Categorical Crossentropy
  - Suitable for multi-class classification at each time step
  - Accepts integer targets (no need for one-hot encoding)
- **Metrics**: Accuracy (character-level prediction accuracy)

### Training Parameters
- **Batch Size**: 16 samples per batch
- **Epochs**: 50
- **Validation Split**: 20% (0.2)
  - Training set: 80% of data
  - Validation set: 20% of data

### Training Process

Both models were trained using the `model.fit()` method:

```python
model.fit(
    [encoder_data, decoder_input_data],
    np.expand_dims(decoder_target_data, -1),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.2
)
```

### Training Monitoring

Training progress was monitored using:
- **Training Loss**: Monitored across epochs
- **Validation Loss**: Monitored across epochs
- **Training Accuracy**: Character-level prediction accuracy on training set
- **Validation Accuracy**: Character-level prediction accuracy on validation set

Visualization plots were generated for:
1. Training vs. Validation Loss
2. Training vs. Validation Accuracy

## Inference Architecture

For prediction, the trained model is split into separate encoder and decoder models:

### Encoder Model (Inference)
- **Input**: Source sequence
- **Output**: Final hidden states [h, c]
- Processes the entire input sequence once to generate context

### Decoder Model (Inference)
- **Inputs**: 
  - Current target character
  - Previous hidden states [h, c]
- **Outputs**: 
  - Character probability distribution
  - Updated hidden states [h, c]
- Processes one character at a time in an autoregressive manner

### Inference Algorithm

1. Clean and normalize input text
2. Tokenize and pad input sequence
3. **Encoding Phase**:
   - Pass input through encoder model
   - Obtain initial hidden states
4. **Decoding Phase**:
   - Initialize with start token `\t`
   - Loop until:
     - End token `\n` is predicted, OR
     - Maximum sequence length is reached
   - At each step:
     - Predict next character using decoder model
     - Use predicted character as next input
     - Update hidden states
5. Concatenate predicted characters
6. Apply Unicode normalization (for Khmer output)

This greedy decoding approach selects the most probable character at each step.

## Model Persistence

### Saved Artifacts

For each model, two files are saved:

#### 1. Model File (.keras)
- **English to Khmer**: `models/khmer_transliterator.keras`
- **Khmer to English**: `models/english_romanizer.keras`
- Contains complete model architecture and trained weights

#### 2. Assets File (.pkl)
- **English to Khmer**: `data/processed/khmer_transliteration_assets.pkl`
- **Khmer to English**: `data/processed/english_romanization_assets.pkl`
- Contains:
  - Source tokenizer (with vocabulary and index mappings)
  - Target tokenizer (with vocabulary and index mappings)
  - Maximum source sequence length
  - Maximum target sequence length

## Evaluation Methods

### Training Metrics

The models were evaluated during training using:

1. **Loss (Sparse Categorical Crossentropy)**:
   - Measures prediction error at each character position
   - Lower values indicate better performance
   - Tracked separately for training and validation sets

2. **Accuracy (Character-Level)**:
   - Percentage of correctly predicted characters
   - Evaluated at each time step
   - Tracked separately for training and validation sets

### Qualitative Testing

Both notebooks include qualitative tests with example inputs:

#### English to Khmer Examples
```python
transliterate("hello")
transliterate("trap")
transliterate("mean luy")
transliterate("kdas")
```

#### Khmer to English Examples
```python
romanize("ហេឡូ")    # hello
romanize("ត្រាប")   # trap
romanize("មានលុយ")  # mean luy
romanize("ខ្ដាស់")  # kdas
```

These manual tests allow visual inspection of model outputs for common words and phrases.

## Limitations and Future Improvements

### Current Limitations

1. **Greedy Decoding**: The inference uses greedy decoding (selecting most probable character at each step) rather than beam search, which may not always produce optimal translations.

2. **No Formal Evaluation Metrics**: The notebooks do not include quantitative evaluation metrics such as:
   - Character Error Rate (CER)
   - Word Error Rate (WER)
   - BLEU score
   - Edit distance

3. **Single Layer LSTM**: The model uses single-layer LSTMs which may limit capacity for complex patterns.

4. **Fixed Vocabulary**: Out-of-vocabulary characters are mapped to `<unk>` token, potentially losing information.

### Suggested Improvements

1. **Implement Beam Search**: Use the defined `BEAM_WIDTH=3` parameter to implement beam search decoding for potentially better quality outputs.

2. **Add Evaluation Suite**: Implement comprehensive evaluation with standard metrics on a held-out test set.

3. **Attention Mechanism**: Add attention layers to help the decoder focus on relevant parts of the input sequence.

4. **Bidirectional Encoder**: Use bidirectional LSTM in the encoder to capture context from both directions.

5. **Deeper Networks**: Experiment with multi-layer LSTMs for increased model capacity.

6. **Data Augmentation**: Apply augmentation techniques to increase training data diversity.

7. **Cross-Validation**: Use k-fold cross-validation for more robust performance estimates.

## Usage

### English to Khmer Transliteration
```python
from tensorflow.keras.models import load_model
import pickle

# Load model and assets
model = load_model("models/khmer_transliterator.keras")
with open("data/processed/khmer_transliteration_assets.pkl", "rb") as f:
    assets = pickle.load(f)

# Use transliterate function
result = transliterate("hello")
```

### Khmer to English Romanization
```python
from tensorflow.keras.models import load_model
import pickle

# Load model and assets
model = load_model("models/english_romanizer.keras")
with open("data/processed/english_romanization_assets.pkl", "rb") as f:
    assets = pickle.load(f)

# Use romanize function
result = romanize("ហេឡូ")
```

## Technical Dependencies

- TensorFlow/Keras (for model implementation)
- Pandas (for data loading)
- NumPy (for numerical operations)
- Matplotlib (for visualization)
- Python unicodedata and re (for text normalization)
