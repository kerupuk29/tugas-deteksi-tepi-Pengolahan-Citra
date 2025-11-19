import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Konfigurasi Halaman
st.set_page_config(page_title="Deteksi Tepi Laplacian", layout="centered")
st.title("Tugas: Deteksi Tepi (Metode Laplacian)")
st.write("Aplikasi deteksi tepi menggunakan operator Laplacian (Derivatif Kedua).")

# --- Sidebar Pengaturan ---
st.sidebar.header("Pengaturan Deteksi")
uploaded_file = st.sidebar.file_uploader("Upload Citra", type=["jpg", "png", "jpeg"])

st.sidebar.info("Laplacian menggunakan ukuran kernel (matriks) untuk menghitung turunan kedua.")

# Pilihan Ukuran Kernel (Harus angka ganjil)
ksize = st.sidebar.selectbox(
    "Ukuran Kernel (Kernel Size)",
    options=[1, 3, 5, 7],
    index=1, # Default pilih 3
    help="Semakin besar kernel, semakin tebal tepian yang terdeteksi, tapi bisa menangkap lebih banyak noise."
)

# --- Panel Utama ---
if uploaded_file is not None:
    # 1. Baca Gambar
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    # 2. Pre-processing
    # Laplacian bekerja pada grayscale
    if len(img_array.shape) == 3:
        gray_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray_img = img_array

    # PENTING: Terapkan Gaussian Blur dulu untuk mengurangi noise
    # Laplacian sangat sensitif terhadap noise, tanpa blur hasilnya akan berantakan
    gray_blur = cv2.GaussianBlur(gray_img, (3, 3), 0)

    # 3. Proses Deteksi Tepi (Laplacian)
    # ddepth=cv2.CV_64F digunakan agar bisa menampung nilai negatif (gradien turun)
    laplacian = cv2.Laplacian(gray_blur, cv2.CV_64F, ksize=ksize)
    
    # Ambil nilai absolut (karena hasil laplacian ada yang negatif) lalu konversi ke uint8
    abs_laplacian = np.absolute(laplacian)
    result_img = np.uint8(abs_laplacian)

    # 4. Tampilkan Hasil
    st.subheader("Perbandingan Hasil")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Citra Asli", use_container_width=True)
        
    with col2:
        st.image(result_img, caption=f"Hasil Laplacian (ksize={ksize})", use_container_width=True)
        
    st.divider()
    st.markdown("""
    **Cara kerja:**
    1. Citra diubah ke Grayscale.
    2. Dihaluskan dengan *Gaussian Blur* untuk mengurangi noise.
    3. Dihitung *Laplacian* (turunan kedua) untuk menemukan perubahan intensitas yang tajam (tepi).
    """)

else:
    st.info("Silakan upload gambar untuk memulai.")
