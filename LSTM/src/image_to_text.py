from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
from .config import MODEL_NAME   


_processor = BlipProcessor.from_pretrained(MODEL_NAME)
_model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME)
_device = "cuda" if torch.cuda.is_available() else "cpu"
_model = _model.to(_device)
_model.eval()
print(f"[image_to_text] BLIP model loaded on device: {_device}")


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────
def caption_image(pil_image: Image.Image) -> str:
    """
    Generate a natural-language caption for the given PIL image.

    Args:
        pil_image: A PIL.Image.Image object (RGB or RGBA).

    Returns:
        A string containing the generated caption.
    """
    if pil_image is None:
        return ""

    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    inputs = _processor(images=pil_image, return_tensors="pt").to(_device)

    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=64,
            num_beams=4,
            early_stopping=True,
        )

    caption = _processor.decode(output_ids[0], skip_special_tokens=True)
    return caption.strip()
