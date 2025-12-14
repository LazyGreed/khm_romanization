import os
import re
import pickle
import unicodedata
import numpy as np
import gradio as gr
import tensorflow as tf

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input

# Constants
EMBED_DIM = 32
LSTM_UNITS = 64

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# LSTM Model paths (from notebooks 01 & 02)
LSTM_ENG_TO_KHM_MODEL_PATH = os.path.join(BASE_DIR, "models", "khmer_transliterator.keras")
LSTM_KHM_TO_ENG_MODEL_PATH = os.path.join(BASE_DIR, "models", "english_romanizer.keras")
LSTM_ENG_TO_KHM_ASSETS_PATH = os.path.join(BASE_DIR, "data", "processed", "khmer_transliteration_assets.pkl")
LSTM_KHM_TO_ENG_ASSETS_PATH = os.path.join(BASE_DIR, "data", "processed", "english_romanization_assets.pkl")

# Transformer Model paths (from notebooks 04 & 05)
TRANSFORMER_ENG_TO_KHM_MODEL_PATH = os.path.join(BASE_DIR, "models", "transformer_eng2khm.keras")
TRANSFORMER_KHM_TO_ENG_MODEL_PATH = os.path.join(BASE_DIR, "models", "transformer_romanizer.keras")
TRANSFORMER_ENG_TO_KHM_ASSETS_PATH = os.path.join(BASE_DIR, "data", "processed", "transformer_eng2khm_assets.pkl")
TRANSFORMER_KHM_TO_ENG_ASSETS_PATH = os.path.join(BASE_DIR, "data", "processed", "transformer_romanization_assets.pkl")

class TransformerModel:
    """Wrapper for transformer transliteration models"""
    
    def __init__(self, model_path, assets_path):
        self.load_assets(assets_path)
        self.model = tf.keras.models.load_model(model_path)
    
    def load_assets(self, assets_path):
        with open(assets_path, "rb") as f:
            assets = pickle.load(f)
        
        # Store tokenizers and max lengths
        for key, value in assets.items():
            setattr(self, key, value)


class TransliteratorModel:
    """Wrapper for LSTM seq2seq transliteration models"""
    
    def __init__(self, model_path, assets_path, lstm_units=64):
        self.lstm_units = lstm_units
        self.load_assets(assets_path)
        self.load_model(model_path)
    
    def load_assets(self, assets_path):
        with open(assets_path, "rb") as f:
            assets = pickle.load(f)
        
        # Store tokenizers and max lengths
        for key, value in assets.items():
            setattr(self, key, value)
    
    def load_model(self, model_path):
        # Load the full model
        model = load_model(model_path)
        
        # Create encoder model
        encoder_inputs = model.input[0]
        encoder_outputs, state_h, state_c = model.get_layer("encoder_lstm").output
        self.encoder_model = Model(encoder_inputs, [state_h, state_c])
        
        # Create decoder model
        decoder_inputs = model.input[1]
        decoder_state_input_h = Input(shape=(self.lstm_units,))
        decoder_state_input_c = Input(shape=(self.lstm_units,))
        decoder_states = [decoder_state_input_h, decoder_state_input_c]
        
        decoder_outputs, state_h, state_c = model.get_layer("decoder_lstm")(
            model.get_layer("decoder_embedding")(decoder_inputs),
            initial_state=decoder_states
        )
        decoder_outputs = model.get_layer("decoder_dense")(decoder_outputs)
        self.decoder_model = Model(
            [decoder_inputs] + decoder_states,
            [decoder_outputs, state_h, state_c]
        )


class BiDirectionalTransliterator:
    """Handles bidirectional transliteration between English and Khmer"""
    
    def __init__(self, model_type="lstm"):
        self.model_type = model_type
        self.load_models(model_type)
    
    def load_models(self, model_type):
        """Load models based on selected type"""
        if model_type == "lstm":
            # Load LSTM models
            self.eng_to_khm = TransliteratorModel(
                LSTM_ENG_TO_KHM_MODEL_PATH, 
                LSTM_ENG_TO_KHM_ASSETS_PATH, 
                LSTM_UNITS
            )
            self.khm_to_eng = TransliteratorModel(
                LSTM_KHM_TO_ENG_MODEL_PATH, 
                LSTM_KHM_TO_ENG_ASSETS_PATH, 
                LSTM_UNITS
            )
        else:
            # Load Transformer models
            self.eng_to_khm = TransformerModel(
                TRANSFORMER_ENG_TO_KHM_MODEL_PATH,
                TRANSFORMER_ENG_TO_KHM_ASSETS_PATH
            )
            self.khm_to_eng = TransformerModel(
                TRANSFORMER_KHM_TO_ENG_MODEL_PATH,
                TRANSFORMER_KHM_TO_ENG_ASSETS_PATH
            )
    
    def detect_language(self, text):
        """Detect if text is Khmer or English"""
        # Check for Khmer Unicode characters
        khmer_chars = re.findall(r'[\u1780-\u17FF]', text)
        english_chars = re.findall(r'[a-zA-Z]', text)
        
        if len(khmer_chars) > len(english_chars):
            return "khmer"
        else:
            return "english"
    
    def transliterate_eng_to_khm(self, text):
        """Transliterate English to Khmer"""
        # Clean input
        text = str(text).strip()
        text = re.sub(r"[^a-z]", "", text.lower())
        
        if not text:
            return ""
        
        if self.model_type == "lstm":
            return self._lstm_eng_to_khm(text)
        else:
            return self._transformer_eng_to_khm(text)
    
    def _lstm_eng_to_khm(self, text):
        """LSTM-based English to Khmer transliteration"""
        # Encode input
        seq = self.eng_to_khm.eng_tokenizer.texts_to_sequences([text])
        encoder_input = pad_sequences(
            seq, 
            maxlen=self.eng_to_khm.max_eng_len, 
            padding='post'
        )
        states = self.eng_to_khm.encoder_model.predict(encoder_input, verbose=0)
        
        # Decode output
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = self.eng_to_khm.khm_tokenizer.word_index['\t']
        stop_condition = False
        decoded_chars = []
        
        while not stop_condition:
            output_tokens, h, c = self.eng_to_khm.decoder_model.predict(
                [target_seq] + states, 
                verbose=0
            )
            char_index = np.argmax(output_tokens[0, -1, :])
            char = self.eng_to_khm.khm_tokenizer.index_word.get(char_index, '')
            
            if char == '\n' or len(decoded_chars) >= self.eng_to_khm.max_khm_len + 1:
                stop_condition = True
            else:
                decoded_chars.append(char)
                target_seq[0, 0] = char_index
                states = [h, c]
        
        return unicodedata.normalize('NFC', ''.join(decoded_chars))
    
    def _transformer_eng_to_khm(self, text):
        """Transformer-based English to Khmer transliteration"""
        # Encode input
        eng_seq = self.eng_to_khm.eng_tokenizer.texts_to_sequences([text])
        encoder_input = pad_sequences(eng_seq, maxlen=self.eng_to_khm.max_eng_len, padding='post')
        
        # Initialize decoder input with start token
        decoder_input = np.zeros((1, self.eng_to_khm.max_khm_len + 1), dtype=np.int32)
        decoder_input[0, 0] = self.eng_to_khm.khm_tokenizer.word_index['\t']
        
        # Decode character by character
        decoded_chars = []
        
        for i in range(self.eng_to_khm.max_khm_len):
            # Predict next character
            predictions = self.eng_to_khm.model.predict([encoder_input, decoder_input], verbose=0)
            
            # Get the character at current position
            char_index = np.argmax(predictions[0, i, :])
            
            # Convert index to character
            if char_index == 0:
                # padding token, skip
                break
            
            char = self.eng_to_khm.khm_tokenizer.index_word.get(char_index, '')
            
            # Stop if end token
            if char == '\n':
                break
            
            # Skip special tokens in output
            if char not in ['\t', '<unk>']:
                decoded_chars.append(char)
            
            # Update decoder input for next position
            if i + 1 < self.eng_to_khm.max_khm_len + 1:
                decoder_input[0, i + 1] = char_index
        
        return unicodedata.normalize('NFC', ''.join(decoded_chars))
    
    def transliterate_khm_to_eng(self, text):
        """Transliterate Khmer to English"""
        # Clean input
        text = str(text).strip()
        text = re.sub(r"[^\u1780-\u17FF]", "", text)
        text = unicodedata.normalize('NFC', text)
        
        if not text:
            return ""
        
        if self.model_type == "lstm":
            return self._lstm_khm_to_eng(text)
        else:
            return self._transformer_khm_to_eng(text)
    
    def _lstm_khm_to_eng(self, text):
        """LSTM-based Khmer to English transliteration"""
        # Encode input
        seq = self.khm_to_eng.khm_tokenizer.texts_to_sequences([text])
        encoder_input = pad_sequences(
            seq, 
            maxlen=self.khm_to_eng.max_khm_len, 
            padding='post'
        )
        states = self.khm_to_eng.encoder_model.predict(encoder_input, verbose=0)
        
        # Decode output
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = self.khm_to_eng.eng_tokenizer.word_index['\t']
        stop_condition = False
        decoded_chars = []
        
        while not stop_condition:
            output_tokens, h, c = self.khm_to_eng.decoder_model.predict(
                [target_seq] + states, 
                verbose=0
            )
            char_index = np.argmax(output_tokens[0, -1, :])
            char = self.khm_to_eng.eng_tokenizer.index_word.get(char_index, '')
            
            if char == '\n' or len(decoded_chars) >= self.khm_to_eng.max_eng_len + 1:
                stop_condition = True
            else:
                decoded_chars.append(char)
                target_seq[0, 0] = char_index
                states = [h, c]
        
        return ''.join(decoded_chars)
    
    def _transformer_khm_to_eng(self, text):
        """Transformer-based Khmer to English transliteration"""
        # Encode input
        khm_seq = self.khm_to_eng.khm_tokenizer.texts_to_sequences([text])
        encoder_input = pad_sequences(khm_seq, maxlen=self.khm_to_eng.max_khm_len, padding='post')
        
        # Initialize decoder input with start token
        decoder_input = np.zeros((1, self.khm_to_eng.max_eng_len + 1), dtype=np.int32)
        decoder_input[0, 0] = self.khm_to_eng.eng_tokenizer.word_index['\t']
        
        # Decode character by character
        decoded_chars = []
        
        for i in range(self.khm_to_eng.max_eng_len):
            # Predict next character
            predictions = self.khm_to_eng.model.predict([encoder_input, decoder_input], verbose=0)
            
            # Get the character at current position
            char_index = np.argmax(predictions[0, i, :])
            
            # Convert index to character
            if char_index == 0:
                # padding token, skip
                break
            
            char = self.khm_to_eng.eng_tokenizer.index_word.get(char_index, '')
            
            # Stop if end token
            if char == '\n':
                break
            
            # Skip special tokens in output
            if char not in ['\t', '<unk>']:
                decoded_chars.append(char)
            
            # Update decoder input for next position
            if i + 1 < self.khm_to_eng.max_eng_len + 1:
                decoder_input[0, i + 1] = char_index
        
        return ''.join(decoded_chars)
    
    def auto_transliterate(self, text):
        """Automatically detect language and transliterate"""
        if not text or not text.strip():
            return ""
        
        language = self.detect_language(text)
        
        if language == "khmer":
            result = self.transliterate_khm_to_eng(text)
            detected = f"Khmer → English ({self.model_type.upper()})"
        else:
            result = self.transliterate_eng_to_khm(text)
            detected = f"English → Khmer ({self.model_type.upper()})"
        
        return result, detected


# Initialize the transliterator with LSTM by default
print("Loading LSTM models...")
transliterator = BiDirectionalTransliterator(model_type="lstm")
print("LSTM models loaded successfully!")


# Gradio interface function
def translate_text(input_text, model_type):
    """Main function for Gradio interface"""
    global transliterator
    
    # Reload models if model type changed
    if transliterator.model_type != model_type:
        print(f"Switching to {model_type.upper()} models...")
        transliterator = BiDirectionalTransliterator(model_type=model_type)
        print(f"{model_type.upper()} models loaded successfully!")
    
    if not input_text or not input_text.strip():
        return "", "No input detected"
    
    result, detected = transliterator.auto_transliterate(input_text)
    return result, detected


# Create Gradio interface
with gr.Blocks(title="Khmer ⇄ English Romanization") as demo:
    gr.Markdown(
        """
        # 🌐 Khmer ⇄ English Romanization
        
        Enter text in **English** or **Khmer** and the app will automatically detect 
        the language and transliterate it to the other language.
        
        Choose between **LSTM** or **Transformer** models.
        
        ### Examples:
        - English → Khmer: `hello`, `trap`, `kdas`
        - Khmer → English: `ហេឡូ`, `ត្រាប`, `ខ្ដាស់`
        """
    )
    
    with gr.Row():
        model_selector = gr.Radio(
            choices=["lstm", "transformer"],
            value="lstm",
            label="Model Type",
            info="Select the model architecture to use"
        )
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                label="Input Text",
                placeholder="Type in English or Khmer...",
                lines=3
            )
            translate_btn = gr.Button("Romanize", variant="primary")
        
        with gr.Column():
            output_text = gr.Textbox(
                label="Output",
                lines=3,
                interactive=False
            )
            detection_info = gr.Textbox(
                label="Detection Info",
                interactive=False
            )
    
    # Examples
    gr.Examples(
        examples=[
            ["hello"],
            ["trap"],
            ["kdas"],
            ["ហេឡូ"],
            ["ត្រាប"],
            ["ខ្ដាស់"],
        ],
        inputs=input_text
    )
    
    # Connect the button
    translate_btn.click(
        fn=translate_text,
        inputs=[input_text, model_selector],
        outputs=[output_text, detection_info]
    )
    
    # Also trigger on Enter key
    input_text.submit(
        fn=translate_text,
        inputs=[input_text, model_selector],
        outputs=[output_text, detection_info]
    )


# Launch the app
if __name__ == "__main__":
    demo.launch(share=False)
