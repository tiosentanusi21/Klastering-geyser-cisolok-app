import streamlit as st
st.set_page_config(page_title="Klastering Geyser Cisolok", layout="wide")  # ⬅️ HARUS di baris ini

from streamlit_option_menu import option_menu
from tio import (
    beranda,
    tabel,
    visual,
    metode,
    kesimpulan
)

selected = option_menu(
    menu_title=None,
    options=["Beranda", "Tabel", "Visual", "Metode", "Kesimpulan"],
    icons=["house", "table", "bar-chart", "book", "clipboard-check"],
    orientation="horizontal",
    default_index=0
)

if selected == "Beranda":
    beranda.show()
elif selected == "Tabel":
    tabel.show()
elif selected == "Visual":
    visual.show()
elif selected == "Metode":
    metode.show()
elif selected == "Kesimpulan":
    kesimpulan.show()
