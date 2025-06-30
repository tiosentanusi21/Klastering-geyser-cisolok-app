import streamlit as st

def show():
    st.title("🔍 Metode Klastering yang Digunakan")

    st.success("Analisis klaster dilakukan menggunakan metode K-Means Clustering.")

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
    <strong>📌 K-Means Clustering</strong><br>
    Metode K-Means digunakan untuk mengelompokkan data responden ke dalam beberapa klaster berdasarkan kesamaan karakteristik. Metode ini efektif untuk segmentasi data karena mampu mengelompokkan data berdasarkan jarak antar titik dalam ruang multidimensi.
    </div>

    <div class='highlight'>
    <strong>⚙️ Langkah-langkah Analisis</strong><br>
    1. Preprocessing data (membersihkan dan menormalkan data)<br>
    2. Menentukan jumlah klaster optimal (menggunakan Elbow & Silhouette Score)<br>
    3. Menerapkan algoritma K-Means untuk membentuk klaster<br>
    4. Visualisasi hasil dan interpretasi klaster
    </div>

    <div class='highlight'>
    <strong>📈 Evaluasi Model</strong><br>
    Model dievaluasi menggunakan Silhouette Score, Calinski-Harabasz Index, dan Davies-Bouldin Index untuk memastikan kualitas pengelompokan yang optimal.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='recommend-box'>
    <b>💡 Catatan Penting:</b>
    <ul>
    <li>🔎 Pemilihan jumlah klaster sangat krusial agar hasil pengelompokan bermakna.</li>
    <li>📊 Visualisasi seperti Elbow Curve dan PCA membantu memahami struktur data.</li>
    <li>🎯 Evaluasi model dilakukan untuk memastikan pembagian klaster efektif dan valid.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
