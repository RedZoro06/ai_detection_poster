"""
Detektor Poster AI vs Non-AI — aplikasi web (Streamlit).

Model: CNN pretrained ImageNet (dibaca dari ringkasan_hasil.json, default DenseNet121
dipotong di pool3_relu) dengan classification head yang dilatih pada 198 poster.

Jalankan lokal :  pip install -r requirements.txt  lalu  streamlit run app.py
Streamlit Cloud:  taruh app.py, requirements.txt, detektor_poster_ai.keras, dan
                  ringkasan_hasil.json di root repo GitHub, Main file path = app.py
"""
import io
import os
import json

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from matplotlib import colormaps

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "detektor_poster_ai.keras")
SUMMARY_PATH = os.path.join(HERE, "ringkasan_hasil.json")

# Harus sama persis dengan preprocessing saat training
W, H = 256, 384
JPEG_QUALITY = 85
CLASS_NAMES = ["Non-AI", "AI"]

st.set_page_config(page_title="Detektor Poster AI", layout="wide")


def load_summary():
    try:
        with open(SUMMARY_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


SUMMARY = load_summary()
BACKBONE = SUMMARY.get("model", "DenseNet121")
CAM_LAYER = SUMMARY.get("lapisan_potong", "pool3_relu")
THRESHOLD = float(SUMMARY.get("ambang", 0.5))


@st.cache_resource(show_spinner="Memuat model...")
def load_model():
    # Import di dalam fungsi supaya TensorFlow hanya dimuat sekali per server
    import keras
    A = keras.applications
    preprocess = {
        "DenseNet121": A.densenet.preprocess_input,
        "VGG16": A.vgg16.preprocess_input,
        "ResNet50": A.resnet50.preprocess_input,
        "InceptionV3": A.inception_v3.preprocess_input,
        "MobileNetV2": A.mobilenet_v2.preprocess_input,
        "EfficientNetB2": A.efficientnet.preprocess_input,
    }[BACKBONE]
    model = keras.models.load_model(MODEL_PATH)
    cam_model = keras.Model(model.inputs, [model.get_layer(CAM_LAYER).output, model.output])
    return model, cam_model, preprocess


def standardize(im: Image.Image) -> Image.Image:
    """Koreksi orientasi EXIF, RGB, center crop rasio 2:3, resize 256 x 384."""
    im = ImageOps.exif_transpose(im).convert("RGB")
    w, h = im.size
    r = W / H
    if w / h > r:
        nw = int(h * r)
        left = (w - nw) // 2
        im = im.crop((left, 0, left + nw, h))
    else:
        nh = int(w / r)
        top = (h - nh) // 2
        im = im.crop((0, top, w, top + nh))
    return im.resize((W, H), Image.LANCZOS)


def jpeg_roundtrip(im: Image.Image) -> Image.Image:
    """Simpan-baca ulang sebagai JPEG kualitas 85, sama seperti dataset bersih."""
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=JPEG_QUALITY)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def predict_proba(arr, model, preprocess) -> float:
    """Probabilitas AI dengan test-time augmentation (gambar asli + flip horizontal)."""
    batch = np.stack([arr, arr[:, ::-1]]).astype("float32")
    return float(model.predict(preprocess(batch), verbose=0).ravel().mean())


def grad_cam(arr, cam_model, preprocess) -> np.ndarray:
    """Peta 0..1 berukuran (H, W): area yang paling mendorong keputusan model."""
    import tensorflow as tf
    x = tf.convert_to_tensor(preprocess(arr[None].astype("float32")))
    with tf.GradientTape() as tape:
        tape.watch(x)
        fmap, p = cam_model(x, training=False)
        p = p[:, 0]
        score = tf.where(p >= THRESHOLD, p, 1 - p)
    grads = tape.gradient(score, fmap)
    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(fmap * weights, axis=-1))[0]
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    return tf.image.resize(cam[..., None], (H, W)).numpy()[..., 0]


def overlay(arr, cam, alpha=0.45) -> Image.Image:
    heat = colormaps["jet"](cam)[..., :3] * 255
    mixed = (1 - alpha) * arr.astype("float32") + alpha * heat
    return Image.fromarray(mixed.clip(0, 255).astype("uint8"))


# ----------------------------------------------------------------------------- UI
st.title("Detektor Poster AI vs Non-AI")
st.write("Unggah sebuah poster, model CNN akan memperkirakan apakah poster tersebut dibuat dengan AI generatif.")

if not os.path.exists(MODEL_PATH):
    st.error("File `detektor_poster_ai.keras` tidak ditemukan. Taruh file model di folder yang sama dengan `app.py`.")
    st.stop()

model, cam_model, preprocess = load_model()

uploaded = st.file_uploader("Pilih gambar poster", type=["jpg", "jpeg", "png", "webp", "bmp"])

if uploaded is not None:
    try:
        image = Image.open(uploaded)
        orig_w, orig_h = ImageOps.exif_transpose(image).size
    except Exception:
        st.error("File tidak bisa dibaca sebagai gambar.")
        st.stop()

    with st.spinner("Menganalisis poster..."):
        arr = np.asarray(jpeg_roundtrip(standardize(image)), dtype=np.uint8)
        p_ai = predict_proba(arr, model, preprocess)
        cam = grad_cam(arr, cam_model, preprocess)

    label = CLASS_NAMES[int(p_ai >= THRESHOLD)]
    confidence = p_ai if label == "AI" else 1 - p_ai

    col1, col2 = st.columns(2)
    with col1:
        st.image(arr, caption="Poster (setelah preprocessing 256 x 384)", width="stretch")
    with col2:
        st.image(overlay(arr, cam), caption="Grad-CAM: merah = area paling berpengaruh", width="stretch")

    st.subheader(f"Prediksi: {label} ({confidence:.0%})")
    st.progress(p_ai, text=f"Probabilitas poster buatan AI: {p_ai:.3f} (ambang {THRESHOLD})")

    if confidence < 0.65:
        st.warning("Model **ragu-ragu** (keyakinan di bawah 65%). Anggap hasil ini tidak meyakinkan.")
    elif confidence < 0.85:
        st.info("Keyakinan model **sedang**.")
    else:
        st.success("Keyakinan model **tinggi**.")

    if orig_w < W or orig_h < H:
        st.warning(
            f"Resolusi gambar kecil ({orig_w} x {orig_h} px, minimal {W} x {H} px). "
            "Gambar buram atau kecil cenderung diprediksi Non-AI, jadi hasil ini kurang bisa dipercaya."
        )

if SUMMARY:
    s = SUMMARY
    st.divider()
    st.markdown(
        f"**Tentang model.** {BACKBONE} pretrained ImageNet (dipotong di lapisan `{CAM_LAYER}`), "
        "dilatih pada 198 poster (100 AI, 98 Non-AI). "
        f"Akurasi cross-validation seluruh data {s.get('akurasi_out_of_fold', 0):.1%} "
        f"(AUC {s.get('auc_out_of_fold', 0):.3f}); akurasi data uji {s.get('akurasi_test', 0):.1%} "
        f"pada {s.get('n_test', 0)} poster.\n\n"
        "**Batasan.** Dataset kecil, poster AI dibuat dengan alat generatif tertentu dan poster Non-AI "
        "diambil dari Pinterest. Hasil pada poster dari generator AI lain atau sumber lain belum teruji, "
        "jadi gunakan hasilnya sebagai indikasi, bukan bukti."
    )
