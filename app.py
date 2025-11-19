import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Konfigurasi Halaman
st.set_page_config(page_title="Deteksi Tepi Citra", layout="centered")
st.title("Tugas: Deteksi Tepi (Edge Detection)")
st.write("Aplikasi sederhana untuk mendeteksi tepi objek pada gambar menggunakan algoritma Canny.")

# --- Sidebar Pengaturan ---
st.sidebar.header("Pengaturan Deteksi")
uploaded_file = st.sidebar.file_uploader("Upload Citra", type=["jpg", "png", "jpeg"])

# Penjelasan singkat tentang Threshold Canny
st.sidebar.info("Algoritma Canny menggunakan dua ambang batas (threshold) untuk menentukan mana yang tepi kuat dan tepi lemah.")

threshold1 = st.sidebar.slider("Threshold Bawah (Min)", 0, 255, 100)
threshold2 = st.sidebar.slider("Threshold Atas (Max)", 0, 255, 200)

# --- Panel Utama ---
if uploaded_file is not None:
    # 1. Baca Gambar
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    # 2. Proses Deteksi Tepi (Canny)
    # Algoritma Canny butuh gambar grayscale, tapi OpenCV biasanya menanganinya. 
    # Namun, konversi eksplisit lebih aman.
    if len(img_array.shape) == 3:
        gray_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray_img = img_array

    # Deteksi Tepi
    edges = cv2.Canny(gray_img, threshold1, threshold2)

    # 3. Tampilkan Hasil
    st.subheader("Perbandingan Hasil")
    
    # Tampilkan berdampingan (Side-by-side)
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Citra Asli", use_container_width=True)
        
    with col2:
        # Canny outputnya hitam putih (grayscale), jadi tidak perlu konversi warna lagi
        st.image(edges, caption="Hasil Deteksi Tepi (Canny)", use_container_width=True)
        
    st.divider()
    st.success("Deteksi tepi selesai! Sesuaikan slider di sebelah kiri jika garis tepi kurang jelas.")

else:
    st.info("Silakan upload gambar untuk memulai.")
