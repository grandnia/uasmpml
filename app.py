import streamlit as st
import joblib
import os
import pandas as pd

# Import class sklearn yang umum dipakai di pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

class MyCustomTransformer:
    def fit(self, X, y=None):
        # fit logic jika perlu, kalau nggak tinggal return self
        return self

    def transform(self, X):
        # contoh transformasi, misal lower case semua string di DataFrame
        X_transformed = X.copy()
        for col in X_transformed.select_dtypes(include=['object']).columns:
            X_transformed[col] = X_transformed[col].str.lower()
        return X_transformed

st.title("Prediksi Profit Menu Restoran")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
pipeline_path = os.path.join(BASE_DIR, "pipeline_rfnew.pkl")

# Load pipeline
pipeline = joblib.load(pipeline_path)

# Input fitur dari user
menu_item = st.text_input('Nama Menu', 'Nasi Goreng')
restaurant_id = st.text_input('ID Restoran', 'R001')
price = st.number_input('Harga Jual per Produk (Rp)', min_value=0.0, value=25000.0, step=1000.0)
menu_category = st.selectbox('Kategori Menu', ['Makanan', 'Minuman', 'Dessert'])
ingredients = st.text_area('Bahan-bahan', 'Nasi, Telur, Ayam, Kecap')

# Buat DataFrame input sesuai kolom yang pipeline harapkan
input_data = pd.DataFrame([{
    'MenuItem': menu_item,
    'RestaurantID': restaurant_id,
    'Price': price,
    'MenuCategory': menu_category,
    'Ingredients': ingredients
}])

if st.button('Prediksi Profit'):
    try:
        prediksi = pipeline.predict(input_data)
        st.success(f"Estimasi profit: Rp {prediksi[0]:,.2f}")
    except Exception as e:
        st.error(f"Terjadi error saat prediksi: {e}")

