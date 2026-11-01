"""AI-based Waste Identification & Disposal Assistant - Streamlit app."""
import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
MODEL_PATH = ROOT / "models" / "model_baseline.keras"
CLASSES_PATH = ROOT / "models" / "class_names.json"
BINS_PATH = APP_DIR / "bin_map.json"
CM_PATH = ROOT / "assets" / "confusion_matrix.png"
IMG_SIZE = 224

st.set_page_config(page_title="AI Waste Assistant", page_icon="♻️")

st.markdown("""
<style>
.hero {padding:1.5rem 1.4rem; border-radius:16px; color:#fff; margin-bottom:1rem;
       background:linear-gradient(135deg,#1b7f4b,#2fa36b 60%,#46b98a);}
.hero .t {font-size:2rem; font-weight:700; line-height:1.2;}
.hero .s {margin-top:.4rem; opacity:.92;}
.result {padding:1rem 1.2rem; border-radius:14px; border-left:8px solid var(--c);
         background:color-mix(in srgb, var(--c) 12%, transparent); margin:.6rem 0;}
.result .n {font-size:1.35rem; font-weight:700;}
.pill {display:inline-block; padding:.15rem .75rem; border-radius:999px; color:#fff;
       background:var(--c); font-weight:600; font-size:.85rem; margin:.3rem 0;}
</style>
<div class="hero"><div class="t">♻️ AI Waste Identification Assistant</div>
<div class="s">Show it one waste item. It tells you what it is and which bin it belongs in.</div></div>
""", unsafe_allow_html=True)


def stream_color(stream: str) -> str:
    s = stream.lower()
    if s.startswith("wet"):
        return "#2e9e5b"
    if s.startswith("special"):
        return "#d64545"
    return "#2f7fd1"


@st.cache_resource(show_spinner="Loading model...")
def load_assets():
    model = tf.keras.models.load_model(MODEL_PATH)
    classes = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
    bins = json.loads(BINS_PATH.read_text(encoding="utf-8"))
    missing = [c for c in classes if c not in bins]
    if missing:
        raise ValueError(f"bin_map.json has no entry for: {missing}")
    return model, classes, bins


def predict(model, img: Image.Image) -> np.ndarray:
    """Same steps as training: RGB, resize to 224x224, raw 0-255 pixels."""
    img = ImageOps.exif_transpose(img).convert("RGB")
    arr = tf.image.resize(np.asarray(img), (IMG_SIZE, IMG_SIZE)).numpy()
    return model.predict(arr[None], verbose=0)[0]


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
    help="Below this confidence the app says it is not sure instead of guessing.")
st.sidebar.caption("Model: MobileNetV2 (transfer learning), 12 waste classes.")

tab_id, tab_perf, tab_about = st.tabs(["🔍 Identify", "📊 How well does it work?", "ℹ️ About"])

with tab_id:
    mode = st.radio("Input", ["Upload a photo", "Use camera"], horizontal=True,
                    label_visibility="collapsed")
    if mode == "Upload a photo":
        source = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    else:
        source = st.camera_input("Take a photo of one waste item")
    with st.expander("Tips for a good photo"):
        st.write("One item, filling most of the frame. Plain background, good light, "
                 "in focus. Cluttered backgrounds confuse the model.")

    if source is None:
        st.info("Upload or capture a photo to get started.")
    else:
        img = Image.open(source)
        st.image(img, caption="Your photo", width=320)
        probs = predict(model, img)
        order = probs.argsort()[::-1]
        top = int(order[0])
        conf = float(probs[top])
        info = bins[classes[top]]

        if conf < threshold:
            st.warning(f"I'm not sure about this one (best guess: {info['label']}, "
                       f"{conf:.0%}). Try a clearer photo: one item, plain background, good light.")
        else:
            c = stream_color(info["stream"])
            st.markdown(
                f"<div class='result' style='--c:{c}'><div class='n'>{info['label']}</div>"
                f"<span class='pill' style='--c:{c}'>{info['stream']}</span>"
                f"<div>{info['tip']}</div></div>", unsafe_allow_html=True)
            st.caption(f"Confidence: {conf:.0%}")

        with st.expander("Top 3 predictions", expanded=conf < threshold):
            for i in order[:3]:
                st.write(f"{bins[classes[int(i)]]['label']} - {probs[int(i)]:.0%}")
                st.progress(float(probs[int(i)]))

with tab_perf:
    a, b, c_, d = st.columns(4)
    a.metric("Test accuracy", "94.1%")
    b.metric("Macro F1", "0.91")
    c_.metric("Images", "15,866")
    d.metric("Classes", "12")
    st.caption("Measured on 2,380 held-out test images the model never saw in training.")
    if CM_PATH.exists():
        st.image(str(CM_PATH), caption="Confusion matrix (rows: true class, columns: predicted)")
    st.subheader("Real-world limits")
    st.write("Dataset photos are cleaner than phone photos, so real-world accuracy is lower. "
             "In our own tests the weak spots were blank paper, steel utensils and tumblers, "
             "plastic bags and clear drinking glasses. Fixing them with our own photos is the next milestone.")

with tab_about:
    st.subheader("How it works")
    st.write("1. You give a photo.  2. It is resized and passed to a MobileNetV2 network "
             "(pretrained on ImageNet, retrained on waste images).  3. The network scores 12 classes.  "
             "4. The best class is mapped to a bin and a disposal tip.  5. If confidence is low, "
             "the app says it is not sure.")
    st.subheader("Bin streams")
    st.markdown("- 🟢 **Wet waste**: food and kitchen waste\n- 🔵 **Dry waste**: paper, plastic, metal, glass, textiles\n"
                "- 🔴 **Special care**: batteries and hazardous items\n"
                "- Sanitary waste is not covered by the classifier.")
    st.subheader("Credits")
    st.write("Dataset: Garbage Classification (12 classes), enhanced version with corrected labels "
             "(original by Mostafa Mohamed, Kaggle). Bin streams follow India's Solid Waste Management Rules, 2026.")
    st.caption("This is an AI suggestion, not an official ruling. Check your local municipal guidelines.")