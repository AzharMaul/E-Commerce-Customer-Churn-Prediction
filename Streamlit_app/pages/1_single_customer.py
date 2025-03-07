# Develop Churn Predictor

# =====================================================================
# Import Library

import streamlit as st
import numpy as np
import pandas as pd
import pickle

# =====================================================================
# Judul
st.title('E-Commerce Single Customer Churn Predictor')
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

# Menambahkan sidebar
st.sidebar.header("Please input customer's, features")

# ====================================================================
# Fungsi untuk menghitung RFM Score
def segment_customer(score):
    if score >= 9:
        return "Best Customers"
    elif score >= 7:
        return "Loyal Customers"
    elif score == 6:
        return "Potential Loyalists"
    elif score == 5:
        return "Needs Attention"
    else:
        return "At Risk"

def compute_rfm(df):
    if df.empty:
        return df

    # Menyesuaikan bins untuk lebih sensitif terhadap pelanggan at risk
    r_bins = [0, 7, 14, 21, 28, 50]  # 0-7 hari = 4, 8-14 hari = 3, 15-30 hari = 2, >30 hari = 1
    f_bins = [0, 2, 5, 9, 14, 16]  # 0-1 order = 1, 1-5 order = 2, 4+ order = 3
    m_bins = [0, 65, 140, 210, 280, 325]  # 0-50 cashback = 1, dst.

    # Memberikan skor berdasarkan bins yang lebih ketat
    df['R_Score'] = pd.cut(df['DaySinceLastOrder'], bins=r_bins, labels=[4, 3, 2, 1, 0], include_lowest=True)
    df['F_Score'] = pd.cut(df['OrderCount'], bins=f_bins, labels=[0, 1, 2, 3, 4], include_lowest=True)
    df['M_Score'] = pd.cut(df['CashbackAmount'], bins=m_bins, labels=[0, 1, 2, 3, 4], include_lowest=True)

    # Mengonversi skor ke integer untuk memastikan hasil numerik
    df[['R_Score', 'F_Score', 'M_Score']] = df[['R_Score', 'F_Score', 'M_Score']].astype(int)

    # Menghitung total RFM Score
    df['RFM_Score'] = df['R_Score'] + df['F_Score'] + df['M_Score']

    # Menentukan segmen pelanggan berdasarkan RFM Score
    df['Customer_Segment'] = df['RFM_Score'].apply(segment_customer)

    return df

# ====================================================================
# Membuat user input

def create_user_input():
    # Numerical Features
    Tenure = st.sidebar.number_input('Tenure (max: 31)', min_value=0.0, max_value=31.0, value=0.0, step=1.0)
    WarehouseToHome = st.sidebar.number_input('Distance from Warehouse to Home (max 36)', min_value=5.0, max_value=36.0, value=5.0, step=1.0)
    HourSpendOnApp = st.sidebar.selectbox('Hour Spend on App', [1, 2, 3, 4, 5]) 
    NumberOfDeviceRegistered = st.sidebar.slider('Number if Device Registered', min_value=1, max_value=6, value=1)
    SatisfactionScore = st.sidebar.slider('Satisfaction Score', min_value=1, max_value=5, value=1)
    NumberOfAddress = st.sidebar.slider('Number of Address', min_value=1, max_value=10, value=1)
    OrderAmountHikeFromlastYear = st.sidebar.number_input('Order Amount Hike From Last Year (max 26)', min_value=11.0, max_value=26.0, value=11.0, step=1.0)
    CouponUsed = st.sidebar.number_input('Coupon Used (max 16)', min_value=0.0, max_value=16.0, value=0.0, step=1.0)
    OrderCount = st.sidebar.number_input('Order Count (max 16)', min_value=1.0, max_value=16.0, value=1.0, step=1.0)
    DaySinceLastOrder = st.sidebar.number_input('Day Since Last Order (max 31)', min_value=0.0, max_value=31.0, value=0.0, step=1.0)
    CashbackAmount = st.sidebar.number_input('Amount of Cashback (max 324.99)', min_value=0.0, max_value=324.99, value=0.0, step=5.0)

    # Categorical Features
    PreferredLoginDevice = st.sidebar.selectbox('Choose Preferred Login Device', ['Mobile Phone', 'Computer'])
    PreferredPaymentMode = st.sidebar.selectbox('Choose Preferred Payment Method', ['Debit Card', 'UPI', 'Credit Card', 'Cash on Delivery', 'E wallet'])
    PreferedOrderCat = st.sidebar.selectbox('Choose Preferred Order Category', ['Laptop & Accessory', 'Mobile', 'Mobile Phone', 'Others', 'Fashion', 'Grocery'])
    MaritalStatus = st.sidebar.selectbox('Choose Marital Status', ['Single', 'Divorced', 'Married'])
    Complain = st.sidebar.selectbox('Complain', ['Yes', 'No'])
    CityTier = st.sidebar.radio('City Tier', [1, 2, 3])

    if Complain == 'Yes':
        Complain = 1
    else:
        Complain = 0

    # Membuat dataframe (nama dan urutannya harus sesuai dengan data train)
    data = pd.DataFrame({
        'Tenure' : Tenure,
        'PreferredLoginDevice': PreferredLoginDevice,
        'CityTier' : CityTier,
        'WarehouseToHome' : WarehouseToHome,
        'PreferredPaymentMode' : PreferredPaymentMode,
        'HourSpendOnApp' : HourSpendOnApp,
        'NumberOfDeviceRegistered': NumberOfDeviceRegistered,
        'PreferedOrderCat' : PreferedOrderCat,
        'SatisfactionScore' : SatisfactionScore,
        'MaritalStatus' : MaritalStatus,
        'NumberOfAddress' : NumberOfAddress,
        'Complain' : Complain,
        'OrderAmountHikeFromlastYear' : OrderAmountHikeFromlastYear,
        'CouponUsed' : CouponUsed,
        'OrderCount' : OrderCount,
        'DaySinceLastOrder' : DaySinceLastOrder,
        'CashbackAmount' : CashbackAmount
    }, index=['value'])

    return data

data_customer = create_user_input()

# ==========================================================================
# Membuat 2 buah container

col1, col2 = st.columns(2)

# bagian kiri (col 1)
with col1:
    # menampilkan customer feature
    st.subheader("Customer's Feature")
    st.write(data_customer.transpose())

# ======================================================================
# Menghitung RFM
rfm_result = compute_rfm(data_customer)

st.subheader("📊 RFM Analysis & Segmentation")
st.write(rfm_result[['DaySinceLastOrder', 'OrderCount', 'CashbackAmount', 'RFM_Score', 'Customer_Segment']])

# ======================================================================
# Membuat prediksi

# Load model
model_loaded = pickle.load(open('xgb_for_churn.sav', 'rb'))
    
# Predict to data
try:
    # Pastikan input sesuai format yang diharapkan oleh model
    kelas = model_loaded.predict(data_customer)
    
    # Menampilkan hasil prediksi dengan tampilan lebih interaktif
    with col2:
        st.subheader('Prediction Result')
        if kelas[0] == 1:
            st.warning('⚠️ **Class 1: This customer will CHURN**')
            st.markdown(
                "<div style='background-color:#ffcccb; padding:10px; border-radius:5px;'>"
                "<b>🚨 Attention:</b> This customer is likely to churn. Consider taking retention measures.</div>",
                unsafe_allow_html=True
            )
        else:
            st.success('✅ **Class 0: This customer will NOT CHURN**')
            st.markdown(
                "<div style='background-color:#d4edda; padding:10px; border-radius:5px;'>"
                "<b>🎉 Great News:</b> This customer is likely to stay loyal. Keep up the good engagement!</div>",
                unsafe_allow_html=True
            )

except Exception as e:
    st.error(f"⚠️ Error in prediction: {e}")
