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
        with open('Streamlit_app/pages/xgb_for_churn.sav', 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None  # Menghindari error jika model gagal dimuat

# Fungsi untuk melakukan prediksi
def predict_churn(model, data):
    predictions = model.predict(data)
    return predictions

# =======================================================================
# Fungsi untuk menangani outlier pada semua kolom numerik
def handle_outliers(df):
    # Menentukan batasan nilai untuk setiap kolom
    limits = {
        'Tenure': (0, 31),
        'CityTier': (1, 3),
        'WarehouseToHome': (5, 36),
        'NumberOfDeviceRegistered': (1, 6),
        'SatisfactionScore': (1, 5),
        'NumberOfAddress': (1, 10),
        'CouponUsed': (0, 16),
        'OrderCount': (1, 16),
        'DaySinceLastOrder': (0, 31),
        'CashbackAmount': (0, 324.99)
    }
    
    # Memeriksa dan menangani outlier berdasarkan rentang yang ditentukan
    for column, (min_value, max_value) in limits.items():
        if column in df.columns:
            df[column] = df[column].apply(lambda x: min(max(x, min_value), max_value))
    
    return df

# Fungsi untuk menghitung RFM
def compute_rfm(df):
    # Pastikan nama kolom tetap sesuai dengan yang ada di dataset
    if not {'DaySinceLastOrder', 'OrderCount', 'CashbackAmount'}.issubset(df.columns):
        st.error("Dataset yang diunggah tidak memiliki kolom yang dibutuhkan: 'DaySinceLastOrder', 'OrderCount', 'CashbackAmount'.")
        return df  # Kembalikan dataset tanpa perubahan jika ada kolom yang hilang

    # Tentukan bins untuk setiap komponen RFM
    bins_recency = [0, 7, 14, 21, 28, 50]  # Recency: makin kecil makin baik
    bins_frequency = [0, 2, 5, 9, 14, 16]  # Frequency: makin besar makin baik
    bins_monetary = [0, 65, 140, 210, 280, 325]  # Monetary: makin besar makin baik

    # Hitung skor RFM
    df['R_Score'] = pd.cut(df['DaySinceLastOrder'], bins=bins_recency, labels=[4, 3, 2, 1, 0], include_lowest=True).astype('int64')
    df['F_Score'] = pd.cut(df['OrderCount'], bins=bins_frequency, labels=[0, 1, 2, 3, 4], include_lowest=True).astype('int64')
    df['M_Score'] = pd.cut(df['CashbackAmount'], bins=bins_monetary, labels=[0, 1, 2, 3, 4], include_lowest=True).astype('int64')

    # Hitung total RFM Score
    df['RFM_Score'] = df['R_Score'] + df['F_Score'] + df['M_Score']

    # Segmentasi pelanggan berdasarkan RFM Score
    def segment_customer(score):
        if score >= 9:
            return "Best Customers"
        elif score >= 7:
            return "Loyal Customers"
        elif score >= 6:
            return "Potential Loyalists"
        elif score >= 5:
            return "Needs Attention"
        else:
            return "At Risk"

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

    data_customer['CustomerID'] = data_customer['CustomerID'].astype(str)

    # Menangani missing values
    if data_customer.isnull().sum().sum() > 0:
        # Penanganan missing values
        for column in data_customer.columns:
            if data_customer[column].isnull().sum() > 0:
                if data_customer[column].dtype == 'object':
                    # For categorical columns, replace missing values with 'Unknown' or mode
                    mode_value = data_customer[column].mode()[0] if not data_customer[column].mode().empty else 'Unknown'
                    data_customer[column].fillna(mode_value, inplace=True)
                else:
                    # For numeric columns, replace missing values with median or mean
                    median_value = data_customer[column].median()
                    # Check if median_value is valid (not NaN)
                    if pd.isna(median_value):
                        median_value = data_customer[column].mean()
                    data_customer[column].fillna(median_value, inplace=True)
        # Menginformasikan bahwa missing values telah ditangani
        st.success("Missing values have been handled successfully.")

    # Menangani outlier
    data_customer = handle_outliers(data_customer)
    st.success("Outliers have been handled successfully.")    
    
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

    # Menghapus kolom 'Churn'
    if 'Churn' in data_customer.columns:
        data_customer = data_customer.drop(columns=['Churn'])

    # Menampilkan data yang diunggah
    st.subheader("Customer's Data from CSV")
    st.write(data_customer)

    # Hitung RFM
    data_customer = compute_rfm(data_customer)

    # Cek apakah model berhasil dimuat sebelum melanjutkan
    if model_loaded is not None:
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
                st.header("Prediction Results with RFM Segmentation")
                st.write(data_customer[['CustomerID', 'DaySinceLastOrder', 'OrderCount', 'CashbackAmount', 'RFM_Score','Churn Prediction', 'Customer_Segment']])
                
                # Menyediakan opsi untuk mengunduh hasil prediksi sebagai CSV
                csv = data_customer.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Prediction Results",
                    data=csv,
                    file_name='churn_predictions.csv',
                    mime='text/csv'
                )

                # Menampilkan segmentasi pelanggan dalam tabel terpisah
                st.header("Customer Segmentation")

                # Membuat tabel untuk setiap segmen
                segments = data_customer['Customer_Segment'].unique()

                for segment in segments:
                    # Menampilkan judul segmen
                    st.subheader(f"Segment: {segment}")
                    
                    # Menampilkan tabel data segmen hanya untuk pelanggan yang churn (Churn Prediction = 1)
                    segment_data = data_customer[(data_customer['Customer_Segment'] == segment) & (data_customer['Churn Prediction'] == 1)]
                    st.write(f"**Tabel Data untuk Segment {segment} yang Churn**:")
                    st.write(segment_data[['CustomerID', 'DaySinceLastOrder', 'OrderCount', 'CashbackAmount', 'RFM_Score','Churn Prediction', 'Customer_Segment']])
                    
                    # Strategi marketing untuk segmen tersebut
                    if segment == 'Best Customers':
                        marketing_strategy = """
                            - Diskon eksklusif tiap bulan + voucher besar.
                            - Layanan tambahan gratis (contoh: free shipping, layanan prioritas).
                        """
                    elif segment == 'Loyal Customers':
                        marketing_strategy = """
                            - Voucher bulanan dengan syarat minimal belanja (misal, beli minimal 2 produk).
                            - Dorong upselling & cross-selling dengan rekomendasi produk yang relevan.
                        """
                    elif segment == 'Potential Loyalists':
                        marketing_strategy = """
                            - Diskon kompetitif.
                            - Follow-up email untuk memahami alasan penurunan transaksi.
                            - Push notification promo produk favorit.
                        """
                    elif segment == 'Needs Attention':
                        marketing_strategy = """
                            - Voucher khusus untuk kembali belanja dengan syarat tertentu.
                            - Survei pelanggan untuk mengetahui alasan menurunnya transaksi.
                            - Retargeting Ads untuk membangkitkan minat belanja lagi.
                        """
                    elif segment == 'At Risk':
                        marketing_strategy = """
                            - Voucher pertama dengan diskon besar untuk menarik minat.
                            - Voucher kedua memiliki syarat minimal belanja 2 produk atau lebih agar mendorong transaksi lebih lanjut.
                        """
                    else:
                        # Jika segmen tidak ditemukan, beri pesan default
                        marketing_strategy = "No marketing strategy defined for this segment."
                    
                    # Menampilkan strategi marketing untuk segmen tersebut
                    st.write(f"**Strategi Marketing untuk Segment {segment}:**")
                    st.write(marketing_strategy)
                    
                    # Menyediakan tombol untuk mengunduh hasil segmen tertentu
                    csv_segment = segment_data.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label=f"Download {segment} Churn Data",
                        data=csv_segment,
                        file_name=f'{segment}_churn_data.csv',
                        mime='text/csv'
                    )
        else:
            st.error("No valid features available for prediction. Please check your dataset.")
    else:
        st.error("Model failed to load. Please check the file and try again.")
else:
    st.write("Upload a CSV file to get predictions.")
