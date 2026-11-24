"""
╔══════════════════════════════════════════════════════════════════╗
║          SISTEM KLASIFIKASI SAMPAH CERDAS - STREAMLIT APP        ║
║  Palet: #CE7F20 | #607938 | #533416                             ║
║  Kategori: Organik · Anorganik · B3                              ║
║  Input  : Kamera Live | Upload Foto                              ║
╚══════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
import matplotlib.pyplot as plt

# ─── Konfigurasi Halaman (harus paling pertama) ─────────────────
st.set_page_config(
    page_title="Klasifikasi Sampah AI",
    page_icon="🗑️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ─── Palet Warna ─────────────────────────────────────────────────
PRIMARY  = "#CE7F20"   # Orange/Gold  — Anorganik
SECONDARY= "#607938"   # Hijau        — Organik
TERTIARY = "#533416"   # Coklat Tua   — aksen/judul
DANGER   = "#DC2626"   # Merah        — B3

# ─── Custom CSS ──────────────────────────────────────────────────
st.markdown(f"""
<style>
  /* ── Header ── */
  .main-header {{
    background: linear-gradient(135deg, {TERTIARY} 0%, {PRIMARY} 100%);
    padding: 2rem 1.5rem;
    border-radius: 14px;
    text-align: center;
    color: white;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 18px rgba(83,52,22,.22);
  }}
  .main-header h1 {{ margin:0; font-size:1.9rem; font-weight:800; letter-spacing:-.5px; }}
  .main-header p  {{ margin:.5rem 0 0; opacity:.88; font-size:1rem; }}

  /* ── Badge ── */
  .result-badge {{
    color: white;
    padding: .6rem 1.8rem;
    border-radius: 50px;
    font-size: 1.35rem;
    font-weight: 700;
    display: inline-block;
    margin: .4rem 0;
    box-shadow: 0 2px 10px rgba(0,0,0,.18);
  }}
  .badge-organik   {{ background:{SECONDARY}; }}
  .badge-anorganik {{ background:{PRIMARY};   }}
  .badge-b3        {{ background:{DANGER};    }}

  /* ── Info boxes ── */
  .info-box    {{ background:#FDF5E8; border-left:4px solid {PRIMARY};   padding:1rem 1.2rem; border-radius:8px; margin:.8rem 0; }}
  .success-box {{ background:#F0F7E8; border-left:4px solid {SECONDARY}; padding:1rem 1.2rem; border-radius:8px; margin:.8rem 0; }}
  .warning-box {{ background:#FEF2F2; border-left:4px solid {DANGER};    padding:1rem 1.2rem; border-radius:8px; margin:.8rem 0; }}

  /* ── Section title ── */
  .section-title {{
    color:{TERTIARY}; font-weight:700; font-size:1.05rem;
    margin:1.2rem 0 .5rem; padding-bottom:.3rem;
    border-bottom:2px solid {PRIMARY};
  }}

  /* ── Tab active ── */
  div.stTabs [data-baseweb="tab-list"] {{ gap:6px; }}
  div.stTabs [data-baseweb="tab"] {{
    border-radius:8px 8px 0 0;
    background:#FDF5E8; font-weight:600; color:{TERTIARY};
  }}
  div.stTabs [aria-selected="true"] {{ background:{PRIMARY}; color:white; }}

  /* ── Metric card ── */
  .metric-card {{
    background:#FDF5E8; border:1px solid #E8D5B0;
    border-radius:10px; padding:.8rem 1rem; text-align:center;
  }}
  .metric-card .val {{ font-size:1.5rem; font-weight:800; color:{PRIMARY}; }}
  .metric-card .lbl {{ font-size:.8rem; color:{TERTIARY}; margin-top:.2rem; }}
</style>
""", unsafe_allow_html=True)

# ─── Konstanta ───────────────────────────────────────────────────
MODEL_PATHS = [
    os.path.join("models", "best_model_final.h5"),
    os.path.join("models", "best_model_final.keras"),
]
CLASS_INFO_PATH = os.path.join("models", "class_names.json")
IMG_SIZE        = (224, 224)

CLASS_COLORS = {"Organik": SECONDARY, "Anorganik": PRIMARY, "B3": DANGER}
BADGE_CLASS  = {"Organik": "badge-organik", "Anorganik": "badge-anorganik", "B3": "badge-b3"}

DISPOSAL_INFO = {
    "Organik": {
        "icon": "🌱",
        "desc": "Dapat dijadikan kompos atau dibuang ke tempat sampah <strong>organik (hijau)</strong>.",
        "tips": [
            "Pisahkan dari sampah anorganik & B3",
            "Olah menjadi pupuk kompos rumahan",
            "Contoh: sisa makanan, daun kering, ranting",
        ],
        "box": "success-box",
    },
    "Anorganik": {
        "icon": "♻️",
        "desc": "Dapat didaur ulang. Pisahkan dan buang ke tempat sampah <strong>anorganik (kuning/orange)</strong>.",
        "tips": [
            "Bersihkan kemasan sebelum dibuang",
            "Pilah berdasarkan jenis: plastik, kertas, kaca, logam",
            "Kumpulkan ke bank sampah terdekat",
        ],
        "box": "info-box",
    },
    "B3": {
        "icon": "⚠️",
        "desc": "<strong>BERBAHAYA!</strong> Jangan campur dengan sampah biasa. Bawa ke tempat pengumpulan <strong>limbah B3</strong>.",
        "tips": [
            "Simpan di wadah tertutup & berlabel",
            "JANGAN dibuang ke saluran air / tanah",
            "Hubungi layanan pengumpulan limbah B3 setempat",
        ],
        "box": "warning-box",
    },
}

# ─── Load Model ──────────────────────────────────────────────────
@st.cache_resource(show_spinner="⏳ Memuat model AI, harap tunggu...")
def load_keras_model():
    for path in MODEL_PATHS:
        if os.path.exists(path):
            try:
                model = tf.keras.models.load_model(path)
                return model, path
            except Exception as e:
                continue
    return None, None


@st.cache_resource
def load_class_info():
    if os.path.exists(CLASS_INFO_PATH):
        with open(CLASS_INFO_PATH) as f:
            return json.load(f)
    # Fallback default
    return {
        "class_names":    ["Anorganik", "B3", "Organik"],
        "best_model":     "CNN",
        "best_scenario":  "N/A",
        "best_f1":        0.0,
        "preprocessing":  "divide255",
    }

# ─── Preprocessing ───────────────────────────────────────────────
def preprocess_image(pil_image: Image.Image, preprocessing: str = "divide255") -> np.ndarray:
    """Resize ke 224×224 dan normalisasi sesuai model."""
    img = pil_image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    if preprocessing == "mobilenet":
        arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    elif preprocessing == "resnet":
        arr = tf.keras.applications.resnet50.preprocess_input(arr)
    else:
        arr = arr / 255.0
    return arr

# ─── Prediksi ────────────────────────────────────────────────────
def predict(model, arr: np.ndarray, class_names: list):
    preds = model.predict(arr, verbose=0)[0]
    idx   = int(np.argmax(preds))
    return class_names[idx], float(preds[idx]), preds

# ─── Tampilkan Hasil ─────────────────────────────────────────────
def show_result(label: str, confidence: float, all_preds: np.ndarray, class_names: list):
    # ── Badge + Progress bar ──
    st.markdown('<p class="section-title">🎯 Hasil Klasifikasi</p>', unsafe_allow_html=True)
    icon  = DISPOSAL_INFO[label]["icon"]
    badge = BADGE_CLASS.get(label, "badge-anorganik")

    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.markdown(
            f'<div style="text-align:center;padding:.6rem">'
            f'<span class="result-badge {badge}">{icon} {label}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with col_b:
        st.markdown(f"**Confidence Score**")
        st.progress(float(confidence))
        st.markdown(f"<span style='font-size:1.5rem;font-weight:800;color:{CLASS_COLORS.get(label, PRIMARY)}'>{confidence*100:.1f}%</span>", unsafe_allow_html=True)

    # ── 3 Metric cards ──
    st.markdown('<p class="section-title">📊 Distribusi Keyakinan Model</p>', unsafe_allow_html=True)
    cols = st.columns(len(class_names))
    sorted_pairs = sorted(zip(class_names, all_preds), key=lambda x: -x[1])
    for col, (cn, prob) in zip(cols, sorted_pairs):
        clr = CLASS_COLORS.get(cn, PRIMARY)
        col.markdown(
            f'<div class="metric-card">'
            f'<div class="val" style="color:{clr}">{prob*100:.1f}%</div>'
            f'<div class="lbl">{cn}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Bar chart horizontal ──
    fig, ax = plt.subplots(figsize=(6, 2.6))
    bar_colors = [CLASS_COLORS.get(cn, PRIMARY) for cn in class_names]
    bars = ax.barh(class_names, [p * 100 for p in all_preds],
                   color=bar_colors, edgecolor="white", height=0.5)
    for bar, prob in zip(bars, all_preds):
        ax.text(prob * 100 + .8, bar.get_y() + bar.get_height() / 2,
                f"{prob*100:.1f}%", va="center", fontsize=10,
                color=TERTIARY, fontweight="bold")
    ax.set_xlim(0, 118)
    ax.set_xlabel("Confidence (%)", color=TERTIARY, fontsize=9)
    ax.tick_params(colors=TERTIARY, labelsize=10)
    for sp in ax.spines.values(): sp.set_edgecolor("#E5E7EB")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FAFAFA")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ── Panduan pembuangan ──
    st.markdown('<p class="section-title">📋 Panduan Pembuangan</p>', unsafe_allow_html=True)
    info      = DISPOSAL_INFO[label]
    tips_html = "".join(f"<li>{t}</li>" for t in info["tips"])
    st.markdown(
        f'<div class="{info["box"]}">'
        f'<strong>{icon} {label}</strong><br>'
        f'{info["desc"]}'
        f'<ul style="margin-top:.5rem;padding-left:1.3rem">{tips_html}</ul>'
        f'</div>',
        unsafe_allow_html=True,
    )

# ─── Sidebar ─────────────────────────────────────────────────────
def render_sidebar(class_info: dict, model_path: str | None):
    with st.sidebar:
        st.markdown(
            f'<div style="background:linear-gradient(135deg,{TERTIARY},{PRIMARY});'
            f'padding:1rem;border-radius:10px;text-align:center;color:white;margin-bottom:1rem">'
            f'<h2 style="margin:0;font-size:1.3rem">🗑️ SampahAI</h2>'
            f'<p style="margin:.3rem 0 0;font-size:.82rem;opacity:.9">Garbage Classification System</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown("### 📖 Tentang Aplikasi")
        st.info(
            "Menggunakan **Deep Learning** untuk mengklasifikasikan sampah menjadi:\n\n"
            "🌱 **Organik** · ♻️ **Anorganik** · ⚠️ **B3**"
        )

        st.markdown("### 🏷️ Kategori Sampah")
        for label, color in CLASS_COLORS.items():
            icon = DISPOSAL_INFO[label]["icon"]
            tip  = DISPOSAL_INFO[label]["tips"][0]
            st.markdown(
                f'<div style="margin-bottom:.7rem">'
                f'<span style="background:{color};color:white;padding:3px 11px;'
                f'border-radius:12px;font-size:.85rem;font-weight:600">{icon} {label}</span>'
                f'<small style="display:block;margin-top:4px;color:#666">{tip}</small>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown("### 🤖 Info Model")
        best_model    = class_info.get("best_model",    "N/A")
        best_scenario = class_info.get("best_scenario", "N/A")
        best_f1       = class_info.get("best_f1",       0.0)
        st.markdown(
            f"- **Arsitektur :** {best_model}\n"
            f"- **Skenario   :** {best_scenario}\n"
            f"- **Macro F1   :** `{best_f1:.4f}`\n"
            f"- **Model file :** `{'✅ Loaded' if model_path else '❌ Not found'}`"
        )

        # Tabel semua hasil jika ada
        all_results = class_info.get("all_results")
        if all_results:
            st.markdown("### 📊 Semua Skenario")
            import pandas as pd
            df = pd.DataFrame(all_results)[["scenario","model","accuracy","f1_macro"]]
            df.columns = ["Kode","Model","Acc","F1"]
            df = df.sort_values("F1", ascending=False).reset_index(drop=True)
            st.dataframe(df.style.format({"Acc": "{:.3f}", "F1": "{:.3f}"}),
                         use_container_width=True, hide_index=True)

        st.markdown("---")
        st.caption("© 2024 Sistem Klasifikasi Sampah\nDeep Learning Final Project")

# ─── Main ────────────────────────────────────────────────────────
def main():
    st.markdown(
        '<div class="main-header">'
        '<h1>🗑️ Sistem Klasifikasi Sampah Cerdas</h1>'
        '<p>Identifikasi jenis sampah menggunakan Kecerdasan Buatan (AI)</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    model, model_path = load_keras_model()
    class_info  = load_class_info()
    class_names = class_info.get("class_names", ["Anorganik", "B3", "Organik"])
    preprocessing = class_info.get("preprocessing", "divide255")

    render_sidebar(class_info, model_path)

    if model is None:
        st.error(
            "⚠️ **Model belum tersedia!**  \n"
            "Jalankan notebook training terlebih dahulu."
        )
        with st.expander("📋 Lihat Langkah-langkah"):
            st.markdown(
                "1. Siapkan Kaggle API key di `~/.kaggle/kaggle.json`\n"
                "2. Install dependensi: `pip install -r requirements.txt`\n"
                "3. Jalankan `garbage_classification.ipynb` dari awal hingga akhir\n"
                "4. Pastikan `models/best_model_final.h5` sudah terbentuk\n"
                "5. Restart aplikasi ini: `streamlit run app.py`"
            )
        return

    # ─── Tab Input ──────────────────────────────────────────────
    tab_cam, tab_upload = st.tabs(["📷  Kamera Live", "📁  Upload Foto"])

    # ══ Tab Kamera ══
    with tab_cam:
        st.markdown('<p class="section-title">📷 Ambil Foto dengan Kamera</p>', unsafe_allow_html=True)
        st.caption("Arahkan kamera ke sampah, lalu klik tombol kamera untuk mengambil foto.")
        cam_img = st.camera_input("Ambil foto", label_visibility="collapsed")

        if cam_img is not None:
            pil_img = Image.open(cam_img)
            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(pil_img, caption="📷 Foto dari Kamera", use_column_width=True)
            with col2:
                with st.spinner("🔍 Menganalisis gambar..."):
                    arr = preprocess_image(pil_img, preprocessing)
                    label, conf, all_preds = predict(model, arr, class_names)
                st.success("✅ Analisis selesai!")
            show_result(label, conf, all_preds, class_names)

    # ══ Tab Upload ══
    with tab_upload:
        st.markdown('<p class="section-title">📁 Upload Foto Sampah</p>', unsafe_allow_html=True)
        st.caption("Unggah foto dalam format JPG, JPEG, PNG, atau WEBP.")
        uploaded = st.file_uploader(
            "Pilih gambar", type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
        )

        if uploaded is not None:
            try:
                pil_img = Image.open(uploaded)
            except Exception as e:
                st.error(f"Gagal membaca gambar: {e}")
                return

            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(pil_img, caption=f"📄 {uploaded.name}", use_column_width=True)
            with col2:
                with st.spinner("🔍 Menganalisis gambar..."):
                    arr = preprocess_image(pil_img, preprocessing)
                    label, conf, all_preds = predict(model, arr, class_names)
                st.success("✅ Analisis selesai!")
            show_result(label, conf, all_preds, class_names)


if __name__ == "__main__":
    main()
