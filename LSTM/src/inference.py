import os
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from .config import MODEL_PATH, TOKENIZER_PATH, LABEL_ENCODER_PATH, MAX_LEN
from .preprocessing import preprocess_text

# ──────────────────────────────────────────────────────────────────────────────
for _path in [MODEL_PATH, TOKENIZER_PATH, LABEL_ENCODER_PATH]:
    if not os.path.exists(_path):
        raise FileNotFoundError(f"[inference] Required asset not found: {_path}")


# ──────────────────────────────────────────────────────────────────────────────
model = load_model(MODEL_PATH)

with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

with open(LABEL_ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)


# ──────────────────────────────────────────────────────────────────────────────
def predict_text(text: str) -> tuple[str, float, dict]:
    processed = preprocess_text(text)

    seq    = tokenizer.texts_to_sequences([processed])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post")

    prediction          = model.predict(padded)          
    predicted_idx       = int(np.argmax(prediction))
    confidence          = float(np.max(prediction))
    predicted_label     = label_encoder.inverse_transform([predicted_idx])[0]

    all_scores = {
        label_encoder.inverse_transform([i])[0]: round(float(p), 4)
        for i, p in enumerate(prediction[0])
    }

    return predicted_label, confidence, all_scores