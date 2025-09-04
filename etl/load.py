import os
import pickle
import gdown
from tensorflow.keras.models import load_model


def load_pipeline(
    model_path="final_models/final_lstm.h5",
    tokenizer_path="final_models/tokenizer.pkl",
    drive_id=None,
    maxlen=120
):
    """
    Load trained deep learning model (.h5) and tokenizer.
    If model is missing and drive_id is provided, download from Google Drive.

    Args:
        model_path (str): Local path to .h5 model file
        tokenizer_path (str): Local path to tokenizer.pkl file
        drive_id (str): Google Drive file ID for the model
        maxlen (int): Sequence length (must match training)

    Returns:
        model, tokenizer, maxlen
    """
    # Download model from Google Drive if not found
    if not os.path.exists(model_path) and drive_id:
        url = f"https://drive.google.com/uc?id={drive_id}"
        print(f"Model not found locally. Downloading from {url} ...")
        gdown.download(url, model_path, quiet=False)

    # Load model
    model = load_model(model_path, compile=False)

    # Load tokenizer
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    print(f"Loaded model from {model_path} and tokenizer from {tokenizer_path}")
    return model, tokenizer, maxlen
