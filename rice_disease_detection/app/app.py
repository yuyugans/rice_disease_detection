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
    layout="centered"
)

st.title("🌾 Identifikasi Penyakit Daun Padi")
st.write(
    "Aplikasi identifikasi penyakit tanaman padi "
    "menggunakan metode Convolutional Neural Network (CNN)."
)

st.divider()

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

with st.expander("Informasi kelas penyakit"):
    st.write("**Blast:** penyakit blas pada tanaman padi.")
    st.write("**Blight:** penyakit hawar daun.")
    st.write("**Tungro:** penyakit tungro pada tanaman padi.")

# ==========================================
# 5. UPLOAD GAMBAR
# ==========================================

st.subheader("Unggah Gambar Daun Padi")

uploaded_file = st.file_uploader(
    "Pilih gambar daun padi",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Gambar yang diunggah",
            use_container_width=True
        )

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

        if st.button("Identifikasi Penyakit", type="primary"):

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

            st.divider()
            st.subheader("Hasil Identifikasi")

            st.success(
                f"Hasil prediksi: {DISEASE_NAMES[predicted_class]}"
            )

            st.metric(
                label="Tingkat Keyakinan Model",
                value=f"{confidence * 100:.2f}%"
            )

            # ==================================
            # 9. MENAMPILKAN PROBABILITAS
            # ==================================

            st.subheader("Probabilitas Setiap Kelas")

            for index, class_name in enumerate(CLASSES):
                probability = float(predictions[0][index])

                st.write(
                    f"**{DISEASE_NAMES[class_name]}** "
                    f"({probability * 100:.2f}%)"
                )

                st.progress(probability)

            st.caption(
                "Catatan: hasil merupakan prediksi model berdasarkan "
                "data pelatihan, bukan diagnosis pasti. "
                "Gunakan foto daun yang jelas dan konsultasikan "
                "dengan ahli pertanian untuk memastikan kondisi tanaman."
            )

    except Exception as error:
        st.error(f"Gambar gagal diproses: {error}")

else:
    st.info("Silakan unggah gambar daun padi untuk memulai identifikasi.")

# ==========================================
# 10. FOOTER
# ==========================================

st.divider()

st.caption(
    "Aplikasi Pengolahan Citra Digital - "
    "Identifikasi Penyakit Tanaman Padi Menggunakan CNN"
)