import os
from dotenv import load_dotenv

load_dotenv() 

# ============================================================
# API Settings
# ============================================================
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise EnvironmentError(
        "HF_TOKEN is not set. "
        "Add it to the .env file ."
    )


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_PATH = os.path.join(BASE_DIR, "assets", "cellula_toxic_data.csv")

MODEL_PATH = os.path.join(BASE_DIR, "assets", "toxic_lstm_model.keras")
TOKENIZER_PATH = os.path.join(BASE_DIR, "assets", "tokenizer.pkl")
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, "assets", "label_encoder.pkl")

MODEL_NAME = "Salesforce/blip-image-captioning-base"

MAX_LEN = 100