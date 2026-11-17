"""
Detektor Poster AI vs Non-AI — aplikasi web (Gradio).

Model: DenseNet121 pretrained ImageNet, dipotong di lapisan pool3_relu, bobot konvolusi
dibekukan, plus classification head yang dilatih pada 198 poster (100 AI, 98 Non-AI).

Jalankan lokal :  pip install -r requirements.txt gradio  lalu  python app.py
Di Colab       :  python app.py --share   (memberi link publik sementara)
Hugging Face   :  unggah seluruh isi folder ini ke sebuah Space (SDK: Gradio)
"""
import io
import os
import sys
import json

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import gradio as gr
import keras
import tensorflow as tf
from matplotlib import colormaps
from PIL import Image, ImageOps

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(HERE, "detektor_poster_ai.keras"))
SUMMARY_PATH = os.path.join(HERE, "ringkasan_hasil.json")

# Harus sama persis dengan preprocessing saat training
W, H = 256, 384
JPEG_QUALITY = 85
CLASS_NAMES = ["Non-AI", "AI"]

try:
    with open(SUMMARY_PATH) as f:
        SUMMARY = json.load(f)
except FileNotFoundError:
    SUMMARY = {}

# Backbone dan lapisan potong dibaca dari ringkasan_hasil.json, jadi app tetap cocok
# kalau notebook memilih arsitektur lain saat dijalankan ulang.
A = keras.applications
PREPROCESS = {
    "DenseNet121": A.densenet.preprocess_input,
    "VGG16": A.vgg16.preprocess_input,
    "ResNet50": A.resnet50.preprocess_input,
    "InceptionV3": A.inception_v3.preprocess_input,
    "MobileNetV2": A.mobilenet_v2.preprocess_input,
    "EfficientNetB2": A.efficientnet.preprocess_input,
}
BACKBONE = SUMMARY.get("model", "DenseNet121")
CAM_LAYER = SUMMARY.get("lapisan_potong", "pool3_relu")
THRESHOLD = float(SUMMARY.get("ambang", 0.5))
preprocess = PREPROCESS[BACKBONE]

model = keras.models.load_model(MODEL_PATH)
cam_model = keras.Model(model.inputs, [model.get_layer(CAM_LAYER).output, model.output])


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


def prepare(im: Image.Image) -> np.ndarray:
    return np.asarray(jpeg_roundtrip(standardize(im)), dtype=np.uint8)


def predict_proba(arr: np.ndarray) -> float:
    """Probabilitas AI dengan test-time augmentation (gambar asli + flip horizontal)."""
    batch = np.stack([arr, arr[:, ::-1]]).astype("float32")
    p = model.predict(preprocess(batch), verbose=0).ravel()
    return float(p.mean())


def grad_cam(arr: np.ndarray) -> np.ndarray:
    """Peta 0..1 berukuran (H, W): area yang paling mendorong keputusan model."""
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


def overlay(arr: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    heat = colormaps["jet"](cam)[..., :3] * 255
    mixed = (1 - alpha) * arr.astype("float32") + alpha * heat
    return Image.fromarray(mixed.clip(0, 255).astype("uint8"))


def analyze(image):
    if image is None:
        raise gr.Error("Unggah gambar poster terlebih dahulu.")
    orig_w, orig_h = ImageOps.exif_transpose(image).size
    arr = prepare(image)
    p_ai = predict_proba(arr)
    label = CLASS_NAMES[int(p_ai >= THRESHOLD)]
    confidence = p_ai if label == "AI" else 1 - p_ai

    if confidence < 0.65:
        note = "Model **ragu-ragu** (keyakinan di bawah 65%). Anggap hasil ini tidak meyakinkan."
    elif confidence < 0.85:
        note = "Keyakinan model **sedang**."
    else:
        note = "Keyakinan model **tinggi**."
    if orig_w < W or orig_h < H:
        note += (
            f"\n\n⚠️ Resolusi gambar kecil ({orig_w} x {orig_h} px, minimal {W} x {H} px). "
            "Gambar buram atau kecil cenderung diprediksi Non-AI, jadi hasil ini kurang bisa dipercaya."
        )

    text = (
        f"### Prediksi: {label} ({confidence:.0%})\n"
        f"Probabilitas poster buatan AI: **{p_ai:.3f}** (ambang {THRESHOLD}).\n\n"
        f"{note}\n\n"
        "Grad-CAM menyorot area poster yang paling memengaruhi keputusan model "
        "(merah = paling berpengaruh)."
    )
    probs = {"AI": p_ai, "Non-AI": 1 - p_ai}
    return probs, overlay(arr, grad_cam(arr)), text


def about_text() -> str:
    if not SUMMARY:
        return ""
    s = SUMMARY
    return (
        "**Tentang model.** "
        f"{s.get('model', 'CNN')} pretrained ImageNet (dipotong di lapisan `{s.get('lapisan_potong', CAM_LAYER)}`), "
        "dilatih pada 198 poster (100 AI, 98 Non-AI). "
        f"Akurasi cross-validation seluruh data {s.get('akurasi_out_of_fold', 0):.1%} "
        f"(AUC {s.get('auc_out_of_fold', 0):.3f}); akurasi data uji {s.get('akurasi_test', 0):.1%} "
        f"pada {s.get('n_test', 0)} poster.\n\n"
        "**Batasan.** Dataset kecil, poster AI dibuat dengan alat generatif tertentu dan poster Non-AI "
        "diambil dari Pinterest. Hasil pada poster dari generator AI lain atau sumber lain belum teruji, "
        "jadi gunakan hasilnya sebagai indikasi, bukan bukti."
    )


EXAMPLES_DIR = os.path.join(HERE, "examples")
examples = (
    [[os.path.join(EXAMPLES_DIR, f)] for f in sorted(os.listdir(EXAMPLES_DIR))
     if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
    if os.path.isdir(EXAMPLES_DIR) else []
)

with gr.Blocks(title="Detektor Poster AI") as demo:
    gr.Markdown(
        "# Detektor Poster AI vs Non-AI\n"
        "Unggah sebuah poster, model CNN akan memperkirakan apakah poster tersebut dibuat dengan AI generatif."
    )
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="pil", label="Poster", height=460)
            btn = gr.Button("Deteksi", variant="primary")
            if examples:
                gr.Examples(examples=examples, inputs=inp, label="Contoh poster")
        with gr.Column():
            out_label = gr.Label(num_top_classes=2, label="Hasil")
            out_text = gr.Markdown()
            out_cam = gr.Image(type="pil", label="Grad-CAM", height=460)
    gr.Markdown(about_text())

    btn.click(analyze, inputs=inp, outputs=[out_label, out_cam, out_text], api_name="predict")
    inp.upload(analyze, inputs=inp, outputs=[out_label, out_cam, out_text], api_name=False)

if __name__ == "__main__":
    demo.launch(share="--share" in sys.argv)
