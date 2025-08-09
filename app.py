import streamlit as st
import joblib
import pandas as pd
import os

st.title("Prediksi Profit Menu Restoran")

# Path file pipeline
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
pipeline_path = os.path.join(BASE_DIR, "pipeline_rf.pkl")

# Load pipeline
pipeline = joblib.load(pipeline_path)

# Input fitur (pastikan sama dengan fitur saat training)
menu_item = st.text_input('Nama Menu', 'Nasi Goreng')
restaurant_id = st.text_input('ID Restoran', 'R001')
price = st.number_input('Harga Jual per Produk (Rp)', min_value=0.0, value=25000.0, step=1000.0)
menu_category = st.selectbox('Kategori Menu', ['Makanan', 'Minuman', 'Dessert'])
ingredients = st.text_area('Bahan-bahan', 'Nasi, Telur, Ayam, Kecap')

# Buat DataFrame sesuai urutan & nama kolom training
input_data = pd.DataFrame([{
    'MenuItem': menu_item,
    'RestaurantID': restaurant_id,
    'Price': price,
    'MenuCategory': menu_category,
    'Ingredients': ingredients
}])

# Prediksi
if st.button('Prediksi Profit'):
    try:
        prediksi = pipeline.predict(input_data)
        st.success(f"Estimasi profit: Rp {prediksi[0]:,.2f}")
    except Exception as e:
        st.error(f"Terjadi error saat prediksi: {e}")
