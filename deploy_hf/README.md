---
title: Detektor Poster AI
emoji: 🖼️
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 6.32.0
python_version: "3.11"
app_file: app.py
pinned: false
---

# Detektor Poster AI vs Non-AI

Aplikasi web untuk mendeteksi apakah sebuah poster dibuat dengan AI generatif.
Model: DenseNet121 pretrained ImageNet (dipotong di lapisan `pool3_relu`, bobot konvolusi dibekukan)
dengan classification head yang dilatih pada 198 poster (100 AI, 98 Non-AI).
Setiap prediksi disertai Grad-CAM yang menunjukkan area poster paling berpengaruh.

| File | Isi |
|---|---|
| `app.py` | Aplikasi Gradio (preprocessing, prediksi, Grad-CAM) |
| `detektor_poster_ai.keras` | Model hasil training |
| `ringkasan_hasil.json` | Metrik evaluasi yang ditampilkan di halaman |
| `requirements.txt` | Library yang dibutuhkan |
