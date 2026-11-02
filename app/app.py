"""AI-based Waste Identification Assistant - Streamlit app."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

# ---- Paths and constants ---------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
MODEL_PATH = ROOT / "models" / "model_baseline.keras"   # trained model
CLASSES_PATH = ROOT / "models" / "class_names.json"     # class order used in training
BINS_PATH = APP_DIR / "bin_map.json"                    # class -> bin stream + tip
ASSETS = ROOT / "assets"                                # figures saved from Colab
IMG_SIZE = 224                                          # input size MobileNetV2 expects

# Per-class F1 scores measured on the test set (from the Colab evaluation)
F1 = {"clothes": 0.997, "shoes": 0.976, "biological": 0.963, "trash": 0.954,
      "battery": 0.947, "cardboard": 0.932, "paper": 0.930, "green-glass": 0.896,
      "brown-glass": 0.866, "plastic": 0.825, "metal": 0.819, "white-glass": 0.819}

st.set_page_config(page_title="AI Waste Assistant", page_icon=":material/recycling:")

# ---- Styling: a quiet, editorial look (colours come from .streamlit/config.toml)
st.markdown("""
<style>
h1, h2, h3 {font-family: Georgia, 'Times New Roman', serif; font-weight: 600;}
.masthead {border-bottom: 1px solid #d9d3c7; padding-bottom: 1rem; margin-bottom: 1rem;}
.eyebrow {text-transform: uppercase; letter-spacing: .12em; font-size: .75rem;
          color: #2f6b4f; font-weight: 600;}
.title {font-family: Georgia, 'Times New Roman', serif; font-size: 2.1rem;
        line-height: 1.15; margin: .25rem 0; color: #1f2a24;}
.sub {color: #5b6660;}
.card {padding: .8rem 1rem; border: 1px solid #e2dccf; border-left: 5px solid var(--c);
       border-radius: 8px; background: #fff; margin: .5rem 0;}
.card .s {color: var(--c); font-size: .75rem; font-weight: 700;
          text-transform: uppercase; letter-spacing: .07em;}
.card .n {font-weight: 600; font-size: 1.1rem;}
</style>
<div class="masthead"><div class="eyebrow">Mini project &middot; SWM Rules 2026</div>
<div class="title">AI Waste Identification Assistant</div>
<div class="sub">Show it one waste item. It names the item and tells you which bin it belongs in.</div></div>
""", unsafe_allow_html=True)


def stream_color(stream: str) -> str:
    """Muted colour per bin stream; anything else is treated as dry."""
    s = stream.lower()
    for key, col in (("wet", "#3f7d58"), ("special", "#b5483a"), ("sanitary", "#7a5c99")):
        if s.startswith(key):
            return col
    return "#3b6ea5"


def card(info: dict):
    """Draw one bin card (stream, item name, disposal tip)."""
    c = stream_color(info["stream"])
    st.markdown(f"<div class='card' style='--c:{c}'><div class='s'>{info['stream']}</div>"
                f"<div class='n'>{info['label']}</div><div>{info['tip']}</div></div>",
                unsafe_allow_html=True)


def show_asset(name: str, caption: str):
    """Show a saved figure from assets/ if it exists."""
    p = ASSETS / name
    if p.exists():
        st.image(str(p), caption=caption)


@st.cache_resource(show_spinner="Loading model...")
def load_assets():
    """Load model, class names and bin map once; check every class has a bin entry."""
    model = tf.keras.models.load_model(MODEL_PATH)
    classes = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
    bins = json.loads(BINS_PATH.read_text(encoding="utf-8"))
    missing = [c for c in classes if c not in bins]
    if missing:
        raise ValueError(f"bin_map.json has no entry for: {missing}")
    return model, classes, bins


def predict(model, img: Image.Image) -> np.ndarray:
    """Same preprocessing as training: RGB, 224x224, raw 0-255 pixels."""
    img = ImageOps.exif_transpose(img).convert("RGB")   # fix phone-photo rotation
    arr = tf.image.resize(np.asarray(img), (IMG_SIZE, IMG_SIZE)).numpy()
    return model.predict(arr[None], verbose=0)[0]       # 12 class probabilities


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

tab_id, tab_guide, tab_model, tab_about = st.tabs([
    ":material/search: Identify", ":material/list_alt: Disposal guide",
    ":material/insights: Model and results", ":material/info: About"])

# ---- Tab 1: identify a photo ----------------------------------------------
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
        order = probs.argsort()[::-1]            # classes, most likely first
        top = int(order[0])
        conf = float(probs[top])
        info = bins[classes[top]]

        if conf < threshold:                      # low confidence: do not guess
            st.warning(f"Not sure about this one (best guess: {info['label']}, {conf:.0%}). "
                       "Try a clearer photo: one item, plain background, good light.")
        else:
            card(info)
            st.caption(f"Confidence: {conf:.0%}")

        with st.expander("Top 3 predictions", expanded=conf < threshold):
            for i in order[:3]:
                st.write(f"{bins[classes[int(i)]]['label']} - {probs[int(i)]:.0%}")
                st.progress(float(probs[int(i)]))

# ---- Tab 2: disposal guide (reads bin_map.json) ----------------------------
with tab_guide:
    st.subheader("Disposal guide")
    st.write("Where each item the model knows goes, under the Solid Waste Management Rules, 2026.")
    pick = st.radio("Show", ["All"] + sorted({v["stream"] for v in bins.values()}),
                    horizontal=True)
    for cls in sorted(bins, key=lambda k: (bins[k]["stream"], k)):
        if pick in ("All", bins[cls]["stream"]):
            card(bins[cls])
    st.caption("Sanitary waste (diapers, pads) is a fourth stream under the rules "
               "but is not detected by the classifier.")

# ---- Tab 3: dataset, model, results ---------------------------------------
with tab_model:
    st.subheader("Dataset")
    st.write("Garbage Classification (12 classes), enhanced version with corrected labels. "
             "15,866 images after removing 4 conflicting duplicates, split 70/15/15 with stratification.")
    show_asset("class_distribution.png", "Images per class (note the imbalance)")
    show_asset("samples.png", "One random sample per class")
    st.subheader("Model")
    st.write("MobileNetV2 pretrained on ImageNet and kept frozen, with a small classification "
             "head trained on top. Class weights offset the imbalance; augmentation adds variety.")
    st.subheader("Results on the held-out test set")
    a, b, c, d = st.columns(4)
    a.metric("Accuracy", "94.1%")
    b.metric("Macro F1", "0.91")
    c.metric("Test images", "2,380")
    d.metric("Classes", "12")
    st.caption("Macro F1 counts every class equally, so it is a fairer score than accuracy "
               "when one class (clothes) is a third of the data.")
    st.bar_chart(pd.Series(F1).sort_values(), horizontal=True, color="#3f7d58")
    show_asset("confusion_matrix.png", "Confusion matrix (rows: true class, columns: predicted)")
    st.subheader("Real-world limits")
    st.write("Dataset photos are cleaner than phone photos, so everyday accuracy is lower. In our "
             "own tests the weak spots were blank paper, steel utensils and tumblers, plastic bags "
             "and clear drinking glasses.")

# ---- Tab 4: about and roadmap ---------------------------------------------
with tab_about:
    st.subheader("How it works")
    st.write("1. You give a photo.  2. It is resized and passed to a MobileNetV2 network.  "
             "3. The network scores 12 classes.  4. The best class is mapped to a bin and a "
             "disposal tip.  5. If confidence is low, the app says it is not sure.")
    st.subheader("Bin streams")
    st.markdown(":green[●] **Wet waste**: food and kitchen waste\n\n"
                ":blue[●] **Dry waste**: paper, plastic, metal, glass, textiles\n\n"
                ":red[●] **Special care**: batteries and hazardous items\n\n"
                ":violet[●] **Sanitary waste**: diapers, pads and similar items "
                "(not detected by the classifier)")
    st.subheader("Roadmap")
    st.markdown("**Next**\n"
                "- Collect our own phone photos for the weak spots and add them to training\n"
                "- Rebalance the clothes class and fine-tune the model\n"
                "- Measure the improvement on a separate set of our own photos\n\n"
                "**Later**\n"
                "- Heatmaps showing which part of the photo drove the prediction\n"
                "- Detect several items in one photo\n"
                "- Cover sanitary and more special-care items\n"
                "- A lighter offline version for phones (TensorFlow Lite)\n"
                "- Disposal tips in Hindi")
    st.subheader("Credits")
    st.write("Dataset: Garbage Classification (12 classes), enhanced version with corrected labels "
             "(original by Mostafa Mohamed, Kaggle). Bin streams follow India's Solid Waste Management Rules, 2026.")
    st.caption("This is an AI suggestion, not an official ruling. Check your local municipal guidelines.")