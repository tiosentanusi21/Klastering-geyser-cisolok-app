import streamlit as st

def show():
    st.title("📝 Kesimpulan Hasil Klastering")

    st.success("Analisis klaster selesai dilakukan dengan 3 kelompok utama!")

    st.markdown("""
    <style>
    .highlight {
        background-color: #f2f9ff;
        border-left: 6px solid #2196F3;
        padding: 1em;
        margin-bottom: 1em;
        border-radius: 6px;
        font-size: 16px;
        color: #000;
    }
    .highlight strong {
        color: #0d47a1;
        font-size: 18px;
    }
    .recommend-box {
        background-color: #fff3cd;
        border-left: 6px solid #ffc107;
        padding: 1em;
        border-radius: 6px;
        margin-top: 20px;
        color: #333;
    }
    .recommend-box li {
        padding: 4px 0;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='highlight'>
    <strong>🔷 Klaster 0 - Sangat Baik</strong><br>
    Kelompok ini memiliki skor tertinggi di semua aspek seperti kunjungan wisata, persepsi UNESCO, kontribusi ekonomi, dan strategi pengembangan. Mereka adalah pihak paling mendukung dan sangat potensial untuk dilibatkan sebagai <i>agent of change</i> dalam pengembangan pariwisata.
    </div>

    <div class='highlight'>
    <strong>🟠 Klaster 2 - Baik</strong><br>
    Kelompok ini memiliki persepsi positif namun tidak sekuat Klaster 0. Edukasi dan pelibatan secara bertahap akan membantu mereka menjadi pendukung aktif.
    </div>

    <div class='highlight'>
    <strong>🔻 Klaster 1 - Kurang Baik</strong><br>
    Persepsi terhadap potensi wisata dan kontribusi ekonominya masih rendah. Kelompok ini perlu mendapatkan perhatian lebih melalui kampanye edukatif dan promosi yang intensif.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='recommend-box'>
    <b>💡 Rekomendasi:</b>
    <ul>
    <li>📘 Tingkatkan edukasi masyarakat terhadap manfaat wisata.</li>
    <li>🌟 Libatkan Klaster 0 sebagai duta lokal pariwisata.</li>
    <li>🔁 Bangun komunikasi dua arah dengan Klaster 1.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)