import streamlit as st
import tensorflow as tf
import numpy as np

from pathlib import Path
from PIL import Image

# ==========================================
# 1. KONFIGURASI APLIKASI
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "rice_disease_cnn.keras"

IMAGE_SIZE = (224, 224)

# Urutan kelas harus sama dengan saat training
CLASSES = ["blast", "blight", "tungro"]

# Nama penyakit untuk ditampilkan
DISEASE_NAMES = {
    "blast": "Penyakit Blast (Blas)",
    "blight": "Penyakit Blight (Hawar Daun)",
    "tungro": "Penyakit Tungro"
}

# ==========================================
# 2. PENGATURAN HALAMAN
# ==========================================

st.set_page_config(
    page_title="Identifikasi Penyakit Padi",
    page_icon="🌾",
    layout="wide"
)

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(ellipse at 8% 0%, rgba(196, 220, 162, 0.30), transparent 34%),
            #f7f8f2;
        color: #23352a;
    }
    [data-testid="stHeader"] {
        background: rgba(247, 248, 242, 0.75);
    }
    [data-testid="stMainBlockContainer"] {
        max-width: 1120px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .hero {
        padding: 2.2rem 2.4rem;
        border: 1px solid #dfe8d5;
        border-radius: 24px;
        background: linear-gradient(120deg, #edf4e5 0%, #fbfcf8 72%);
        margin-bottom: 1.5rem;
    }
    .hero-tag {
        display: inline-block;
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
        background: #dcebcf;
        color: #35613b;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    .hero h1 {
        color: #214b31;
        font-size: clamp(2rem, 4vw, 3.1rem);
        line-height: 1.12;
        margin: 0.9rem 0 0.55rem;
    }
    .hero p {
        color: #607064;
        font-size: 1.05rem;
        margin: 0;
        max-width: 680px;
    }
    .section-label {
        color: #527254;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin: 0.25rem 0 0.2rem;
    }
    .disease-card {
        min-height: 118px;
        padding: 1rem 1.1rem;
        border: 1px solid #e1e8dc;
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.8);
    }
    .disease-card span {
        color: #648064;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .disease-card h3 {
        color: #294d34;
        font-size: 1.1rem;
        margin: 0.45rem 0 0.25rem;
    }
    .disease-card p {
        color: #758076;
        font-size: 0.9rem;
        margin: 0;
    }
    [data-testid="stFileUploaderDropzone"] {
        border: 1px dashed #9ab38b;
        border-radius: 14px;
        background: #fbfcf8;
    }
    [data-testid="stImage"] img {
        border-radius: 16px;
    }
    [data-testid="stMetric"] {
        padding: 1rem;
        border: 1px solid #e1e8dc;
        border-radius: 14px;
        background: #ffffff;
    }
    div.stButton > button[kind="primary"] {
        width: 100%;
        min-height: 3rem;
        border-radius: 12px;
        background: #397348;
        border-color: #397348;
        font-weight: 700;
    }
    div.stButton > button[kind="primary"]:hover {
        background: #2e603b;
        border-color: #2e603b;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="hero">
        <span class="hero-tag">🌾 Asisten kesehatan tanaman</span>
        <h1>Kenali penyakit daun padi lebih awal</h1>
        <p>Unggah foto daun untuk mendapatkan prediksi awal berbasis
        Convolutional Neural Network (CNN).</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown('<p class="section-label">Penyakit yang dapat dikenali</p>', unsafe_allow_html=True)
disease_columns = st.columns(3)
disease_cards = [
    ("Blast", "Blas", "Bercak khas yang dapat merusak daun dan batang."),
    ("Blight", "Hawar daun", "Gangguan pada daun yang perlu dipantau sejak dini."),
    ("Tungro", "Tungro", "Penyakit tanaman padi yang berkaitan dengan virus."),
]
for column, (name, description, detail) in zip(disease_columns, disease_cards):
    with column:
        st.markdown(
            f"""
            <div class="disease-card">
                <span>{name}</span>
                <h3>{description}</h3>
                <p>{detail}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

st.write("")

# ==========================================
# 3. MEMUAT MODEL
# ==========================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

if not MODEL_PATH.exists():
    st.error(
        "Model tidak ditemukan. Pastikan file "
        "'rice_disease_cnn.keras' berada di folder models."
    )
    st.stop()

try:
    model = load_model()
except Exception as error:
    st.error(f"Gagal memuat model: {error}")
    st.stop()

# ==========================================
# 4. INFORMASI PENYAKIT
# ==========================================

st.markdown('<p class="section-label">Mulai identifikasi</p>', unsafe_allow_html=True)
upload_column, preview_column = st.columns([1, 1], gap="large")
with upload_column:
    st.subheader("1. Unggah foto")
    uploaded_file = st.file_uploader(
        "Pilih gambar daun padi",
        type=["jpg", "jpeg", "png"],
        help="Gunakan foto daun yang fokus dan memiliki pencahayaan cukup."
    )
    st.caption("Format JPG, JPEG, atau PNG")

if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file).convert("RGB")

        with preview_column:
            st.subheader("2. Pratinjau foto")
            st.image(image, caption=uploaded_file.name, use_container_width=True)

        # ======================================
        # 6. PREPROCESSING GAMBAR
        # ======================================

        resized_image = image.resize(IMAGE_SIZE)

        image_array = np.array(resized_image, dtype=np.float32)

        # Menambahkan dimensi batch
        image_array = np.expand_dims(image_array, axis=0)

        # Normalisasi tidak dilakukan di sini karena
        # model sudah memiliki layer Rescaling(1/255).

        # ======================================
        # 7. PREDIKSI MODEL CNN
        # ======================================

        with upload_column:
            identify_clicked = st.button(
                "✨  Identifikasi penyakit",
                type="primary",
                disabled=uploaded_file is None
            )

        if identify_clicked:

            with st.spinner("Sedang menganalisis gambar..."):

                predictions = model.predict(
                    image_array,
                    verbose=0
                )

                predicted_index = int(np.argmax(predictions[0]))
                predicted_class = CLASSES[predicted_index]
                confidence = float(predictions[0][predicted_index])

            # ==================================
            # 8. MENAMPILKAN HASIL PREDIKSI
            # ==================================

            st.write("")
            st.markdown('<p class="section-label">Hasil analisis</p>', unsafe_allow_html=True)
            st.subheader(DISEASE_NAMES[predicted_class])
            st.write(
                "Prediksi teratas berdasarkan foto yang diunggah. "
                "Periksa tingkat keyakinan dan rincian kelas berikut."
            )

            result_column, note_column = st.columns([1, 2])
            with result_column:
                st.metric(
                    label="Tingkat keyakinan",
                    value=f"{confidence * 100:.2f}%"
                )
            with note_column:
                st.info(
                    "Hasil ini merupakan prediksi model, bukan diagnosis pasti. "
                    "Konsultasikan dengan ahli pertanian untuk memastikan kondisi tanaman."
                )

            # ==================================
            # 9. MENAMPILKAN PROBABILITAS
            # ==================================

            st.subheader("Probabilitas setiap kelas")
            probability_columns = st.columns(3)
            for index, class_name in enumerate(CLASSES):
                probability = float(predictions[0][index])
                with probability_columns[index]:
                    st.write(f"**{DISEASE_NAMES[class_name]}**")
                    st.progress(probability, text=f"{probability * 100:.2f}%")

    except Exception as error:
        st.error(f"Gambar gagal diproses: {error}")

else:
    with preview_column:
        st.subheader("2. Pratinjau foto")
        st.info("Pratinjau gambar akan muncul di sini setelah foto dipilih.")

# ==========================================
# 10. FOOTER
# ==========================================

st.write("")
st.caption(
    "Identifikasi Penyakit Tanaman Padi · CNN | Gunakan foto yang jelas untuk hasil terbaik."
)