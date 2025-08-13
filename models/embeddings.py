from sentence_transformers import SentenceTransformer
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from pathlib import Path

# load CLIP
CLIP_MODEL = "clip-ViT-B-32"
model = SentenceTransformer(CLIP_MODEL)

# Load BLIP
BLIP_MODEL = "Salesforce/blip-image-captioning-base"
processor = BlipProcessor.from_pretrained(BLIP_MODEL)
BLIP_model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL)


def generate_caption(image_path):
    img = Image.open(image_path).convert("RGB")
    inputs = processor(img, return_tensors="pt")
    out = BLIP_model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption


def image_to_vector(filepath):
    """
    filepath: path to image file
    returns: numpy vector
    """
    img = Image.open(filepath).convert("RGB")
    vec = model.encode([img], convert_to_numpy=True,
                       show_progress_bar=False)[0]
    return vec


def text_to_vector(text_or_path):
    """
    Accepts either a text string or path to a text file.
    Returns: numpy vector
    """
    if Path(text_or_path).exists():
        text = Path(text_or_path).read_text(encoding="utf-8")
    else:
        text = text_or_path or ""
    vec = model.encode(text, convert_to_numpy=True, show_progress_bar=False)
    return vec
