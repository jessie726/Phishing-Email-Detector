import pickle
from tensorflow.keras.models import load_model

def load_pipeline(model_path="models/final_lstm.h5", tokenizer_path="models/tokenizer.pkl"):
    """
    Load trained deep learning model (.h5) and tokenizer (.pkl).
    Returns (model, tokenizer, maxlen).
    """
    # Load model
    model = load_model(model_path)

    # Load tokenizer
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    MAXLEN = 120  

    print(f"Loaded model from {model_path} and tokenizer from {tokenizer_path}")
    return model, tokenizer, MAXLEN