import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def show():
    st.title("📊 Eksplorasi Data (EDA)")

    uploaded_file = st.file_uploader("📁 Unggah Dataset Excel", type=["xlsx"])
    
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        st.success("✅ Data berhasil dimuat!")

        # Simpan ke session state
        st.session_state.df = df

        # Tampilkan tabel awal
        st.subheader("🧾 Tabel Data")
        st.dataframe(df.head())

        st.markdown("---")
        st.subheader("📌 Statistik Deskriptif - Usia")
        st.write(df[['Usia']].describe())

        st.markdown("---")
        st.subheader("📌 Distribusi Variabel Kategorikal")
        kategorikal = ['Jenis Kelamin', 'Pendidikan Terakhir', 'Pekerjaan', 'Domisili', 'Status Responden']
        for col in kategorikal:
            st.write(f"**Distribusi {col}:**")
            st.dataframe(df[col].value_counts())

        st.markdown("---")
        st.subheader("📊 Histogram Usia")
        plt.figure(figsize=(8, 5))
        sns.histplot(df['Usia'], bins=10, kde=True, color='skyblue')
        plt.title("Distribusi Usia Responden")
        plt.xlabel("Usia")
        plt.ylabel("Frekuensi")
        plt.grid(True)
        st.pyplot()

        st.markdown("---")
        st.subheader("🍰 Pie Chart - Jenis Kelamin")
        plt.figure(figsize=(5, 5))
        df['Jenis Kelamin'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, colors=['lightblue', 'pink'])
        plt.title("Persentase Jenis Kelamin Responden")
        plt.ylabel("")
        st.pyplot()

        st.markdown("---")
        st.subheader("📊 Barplot Variabel Kategorikal")
        for col in kategorikal:
            plt.figure(figsize=(8, 4))
            sns.countplot(data=df, y=col, order=df[col].value_counts().index, palette="viridis")
            plt.title(f"Distribusi {col}")
            plt.xlabel("Jumlah")
            plt.ylabel(col)
            st.pyplot()

        st.markdown("---")
        st.subheader("🔍 Crosstab Jenis Kelamin vs Pendidikan")
        ct = pd.crosstab(df['Jenis Kelamin'], df['Pendidikan Terakhir'])
        st.dataframe(ct)
    else:
        st.info("Silakan unggah file Excel untuk memulai eksplorasi data.")
