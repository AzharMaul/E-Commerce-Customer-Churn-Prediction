# =======================================================================
# Import Library

import streamlit as st
import pandas as pd
import pickle

from typing import Literal

# =======================================================================
# Judul
st.title('E-Commerce Multiple Customer Churn Predictor')
# Justified Text
st.markdown("""
<div style='text-align: justify;'>
    Welcome to our customer churn prediction app! Designed to help e-commerce businesses
    retain valuable customers. Using the XGBoost Classification, our model analyzes customer
    characteristics to accurately predict churn risk. We focus on the F2 Score metric,
    minimizing false negatives to ensure you don't miss at-risk customers. Transform your
    customer retention approach and make data-driven decisions to boost loyalty and growth!
</div>
""", unsafe_allow_html=True)

# =======================================================================
# Fungsi untuk memuat model dengan penanganan error
@st.cache_resource
def load_model():
    try:
        with open(r'xgb_for_churn.sav', 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None  # Menghindari error jika model gagal dimuat

# Fungsi untuk melakukan prediksi
def predict_churn(model, data):
    predictions = model.predict(data)
    return predictions

# Fungsi untuk menghitung RFM
def compute_rfm(df):
    # Pastikan nama kolom tetap sesuai dengan yang ada di dataset
    if not {'DaySinceLastOrder', 'OrderCount', 'CashbackAmount'}.issubset(df.columns):
        st.error("Dataset yang diunggah tidak memiliki kolom yang dibutuhkan: 'DaySinceLastOrder', 'OrderCount', 'CashbackAmount'.")
        return df  # Kembalikan dataset tanpa perubahan jika ada kolom yang hilang

    # Tentukan bins untuk setiap komponen RFM
    bins_recency = [-1, 7, 14, 30, 31]  # Recency: makin kecil makin baik
    bins_frequency = [0, 3, 7, 12, 16]  # Frequency: makin besar makin baik
    bins_monetary = [0, 50, 150, 250, 325]  # Monetary: makin besar makin baik
    labels = [4, 3, 2, 1]  # Skor: 4 (terbaik) -> 1 (terendah)

    # Hitung skor RFM
    df['R_Score'] = pd.cut(df['DaySinceLastOrder'], bins=bins_recency, labels=labels, include_lowest=True).astype('int64')
    df['F_Score'] = pd.cut(df['OrderCount'], bins=bins_frequency, labels=labels, include_lowest=True).astype('int64')
    df['M_Score'] = pd.cut(df['CashbackAmount'], bins=bins_monetary, labels=labels, include_lowest=True).astype('int64')

    # Hitung total RFM Score
    df['RFM_Score'] = df['R_Score'] + df['F_Score'] + df['M_Score']

    # Segmentasi pelanggan berdasarkan RFM Score
    def segment_customer(score):
        if score >= 10:
            return "Best Customers"
        elif score >= 8:
            return "Loyal Customers"
        elif score >= 6:
            return "Potential Loyalists"
        elif score >= 4:
            return "At Risk"
        else:
            return "Churned Customers"

    df['Customer_Segment'] = df['RFM_Score'].apply(segment_customer)
    
    return df

# =======================================================================
# Menambahkan sidebar dan instruksi untuk mengunggah file
st.sidebar.header("Upload Customer Data CSV")

# Fitur untuk mengunggah file CSV
uploaded_file = st.sidebar.file_uploader("Upload file CSV", type=["csv"])

# Load model
model_loaded = load_model()

# =======================================================================
# Jika file CSV diunggah
if uploaded_file is not None:
    # Membaca file CSV menggunakan pandas
    data_customer = pd.read_csv(uploaded_file)

    # Normalisasi kategori agar sesuai dengan model
    if 'PreferredLoginDevice' in data_customer.columns:
        known_categories = {'Mobile Phone', 'Computer'}
        data_customer['PreferredLoginDevice'] = data_customer['PreferredLoginDevice'].replace({
            'Phone': 'Mobile Phone',
            'Tablet': 'Mobile Phone',
            'Desktop': 'Computer',
            'Laptop': 'Computer'
        })
        data_customer['PreferredLoginDevice'] = data_customer['PreferredLoginDevice'].apply(lambda x: x if x in known_categories else 'Other')

    if 'PreferredPaymentMode' in data_customer.columns:
        known_payment_modes = {'Credit Card', 'Debit Card', 'E wallet', 'UPI', 'Cash on Delivery'}
        data_customer['PreferredPaymentMode'] = data_customer['PreferredPaymentMode'].replace({
            'CC': 'Credit Card',
            'COD': 'Cash on Delivery'
        })
        data_customer['PreferredPaymentMode'] = data_customer['PreferredPaymentMode'].apply(lambda x: x if x in known_payment_modes else x)

    # Menampilkan data yang diunggah
    st.subheader("Customer's Data from CSV")
    st.write(data_customer)

    # Hitung RFM
    data_customer = compute_rfm(data_customer)

    # Cek apakah model berhasil dimuat sebelum melanjutkan
    if model_loaded is not None:
        # Cek apakah ada missing values
        if data_customer.isnull().sum().sum() > 0:
            st.warning("Uploaded data contains missing values. Please clean it before prediction.")
        else:
            # Pastikan hanya menggunakan fitur yang tersedia di dataset
            expected_features = model_loaded.feature_names_in_
            available_features = [f for f in expected_features if f in data_customer.columns]
            
            if len(available_features) < len(expected_features):
                missing_features = set(expected_features) - set(available_features)
                st.warning(f"Some features are missing and will be excluded from prediction: {missing_features}")
            
            # Jika masih ada fitur yang bisa digunakan, lakukan prediksi
            if available_features:
                if st.button('Predict Churn'):
                    predictions = predict_churn(model_loaded, data_customer[available_features])
                    
                    # Menambahkan hasil prediksi ke data
                    data_customer['Churn Prediction'] = predictions
                    
                    # Menampilkan hasil prediksi dan segmentasi pelanggan
                    st.subheader("Prediction Results with RFM Segmentation")
                    st.write(data_customer[['Churn Prediction', 'RFM_Score', 'Customer_Segment']])
                    
                    # Menyediakan opsi untuk mengunduh hasil prediksi sebagai CSV
                    csv = data_customer.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Prediction Results",
                        data=csv,
                        file_name='churn_predictions.csv',
                        mime='text/csv'
                    )
            else:
                st.error("No valid features available for prediction. Please check your dataset.")
    else:
        st.error("Model failed to load. Please check the file and try again.")
else:
    st.write("Upload a CSV file to get predictions.")
