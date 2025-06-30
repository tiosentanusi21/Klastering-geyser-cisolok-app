import streamlit as st
from PIL import Image
import pandas as pd

def show():
    st.markdown("""
    <h1 style='text-align: center;'>Selamat Datang di Aplikasi Klastering Persepsi Wisata</h1>
    <h4 style='text-align: center; color: gray;'>Geyser Cisolok – Kabupaten Sukabumi</h4>
    """, unsafe_allow_html=True)

    st.write("📌 Aplikasi ini bertujuan untuk mengelompokkan persepsi masyarakat terhadap pengembangan pariwisata Geyser Cisolok, menggunakan pendekatan analisis klastering berbasis data survei.")

    # Gambar utama
    image = Image.open("tio/gambar/gesyer.jpg")
    st.image(image, caption='Geyser Cisolok - Potensi Wisata Alam Sukabumi', use_column_width=True)

    # Fitur aplikasi
    st.subheader("✨ Fitur Utama Aplikasi")
    st.markdown("""
    - 📁 Upload dan kelola data persepsi responden
    - 📊 Visualisasi hasil klastering secara interaktif
    - 📌 Informasi metode dan kesimpulan berbasis data
    """)

    # Potensi wisata
    st.subheader("🌋 Potensi Geyser Cisolok")
    potensi = [
        "Fenomena geyser alami satu-satunya di Indonesia",
        "Sumber air panas mengandung belerang tinggi",
        "Daya tarik geowisata berbasis edukasi dan alam",
        "Akses mudah dari kawasan Pantai Pelabuhan Ratu",
        "Berpotensi masuk nominasi Warisan Dunia UNESCO"
    ]
    for item in potensi:
        st.write(f"• {item}")

    # Video dokumenter
    st.subheader("🎥 Video Dokumenter Geyser Cisolok")
    st.video("https://www.youtube.com/watch?v=RkyFpUOqOxM&pp=ygUXZ2V5c2VyIGNpc29sb2sgc3VrYWJ1bWk%3D")

    # Lokasi Peta
    st.subheader("📍 Lokasi Geyser Cisolok di Peta")

    # Titik koordinat geyser Cisolok
    location_data = pd.DataFrame({
        'lat': [-6.989739],  # Lintang
        'lon': [106.492126]  # Bujur
    })

    st.map(location_data, zoom=12)
