"""AI-based Waste Identification & Disposal Assistant - Streamlit app."""
import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR.parent / "models" / "model_baseline.keras"
CLASSES_PATH = APP_DIR.parent / "models" / "class_names.json"
BINS_PATH = APP_DIR / "bin_map.json"
IMG_SIZE = 224


@st.cache_resource(show_spinner="Loading model...")
def load_assets():
    model = tf.keras.models.load_model(MODEL_PATH)
    classes = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
    bins = json.loads(BINS_PATH.read_text(encoding="utf-8"))
    missing = [c for c in classes if c not in bins]
    if missing:
        raise ValueError(f"bin_map.json has no entry for: {missing}")
    return model, classes, bins


def preprocess(img: Image.Image) -> np.ndarray:
    """Same steps as training: RGB, resize to 224x224.
    Pixel rescaling is already inside the model, so raw 0-255 values go in."""
    img = ImageOps.exif_transpose(img).convert("RGB")
    arr = tf.image.resize(np.asarray(img), (IMG_SIZE, IMG_SIZE)).numpy()
    return arr[None]


def predict(model, img: Image.Image) -> np.ndarray:
    return model.predict(preprocess(img), verbose=0)[0]


def main():
    st.set_page_config(page_title="Waste Identification Assistant", page_icon="♻️")
    st.title("♻️ AI Waste Identification & Disposal Assistant")
    st.write(
        "Show the app **one waste item**, ideally on a plain background, and it will "
        "tell you what it is and which bin it belongs in."
    )

    try:
        model, classes, bins = load_assets()
    except FileNotFoundError as e:
        st.error(f"Could not find a required file: {e.filename}")
        st.stop()
    except ValueError as e:
        st.error(str(e))
        st.stop()

    threshold = st.sidebar.slider(
        "Confidence threshold", 0.30, 0.95, 0.60, 0.05,
        help="Below this confidence the app says it is not sure instead of guessing.",
    )
    st.sidebar.caption("Model: MobileNetV2 (transfer learning), 12 waste classes.")

    tab_upload, tab_camera = st.tabs(["Upload a photo", "Use camera"])
    with tab_upload:
        uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    with tab_camera:
        captured = st.camera_input("Take a photo of one waste item")

    source = captured if captured is not None else uploaded
    if source is None:
        st.info("Upload or capture a photo to get started.")
        st.stop()

    img = Image.open(source)
    st.image(img, caption="Your photo", width=320)

    probs = predict(model, img)
    order = probs.argsort()[::-1]
    top = int(order[0])
    conf = float(probs[top])
    info = bins[classes[top]]

    if conf < threshold:
        st.warning(
            f"I'm not sure about this one (best guess: {info['label']}, {conf:.0%}). "
            "Try a clearer photo: one item, plain background, good light."
        )
    else:
        st.success(f"Detected: **{info['label']}** ({conf:.0%} confidence)")
        st.subheader(f"Bin: {info['stream']}")
        st.write(info["tip"])

    with st.expander("Top 3 predictions", expanded=conf < threshold):
        for i in order[:3]:
            st.write(f"{bins[classes[int(i)]]['label']} - {probs[int(i)]:.0%}")
            st.progress(float(probs[int(i)]))

    st.caption(
        "This is an AI suggestion, not an official ruling. Check your local municipal "
        "guidelines. The model was trained on a public dataset and is less accurate "
        "on cluttered real-world photos."
    )


main()