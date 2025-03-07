import streamlit as st

# Set page configuration
st.set_page_config(
    page_title='E-Commerce Customer Churn Prediction',
    layout='wide'
)

# Page title with centered alignment
st.markdown("<h1 style='text-align: center;'>E-Commerce Customer Churn Prediction</h1>", unsafe_allow_html=True)

# Description
st.markdown("""
    <div style='text-align: justify;'>
        Welcome to the E-Commerce Customer Churn Prediction application. 
        Here you have two options to predict whether a customer will churn or not:
        <ul>
            <li><b>Single Customer Prediction:</b> Enter data for a single customer and predict the likelihood of churn.</li>
            <li><b>Multiple Customer Prediction:</b> Upload a CSV file with multiple customer data for bulk predictions.</li>
        </ul>
        Please select a page from the sidebar or click one of the buttons below to get started!
    </div>
""", unsafe_allow_html=True)

# Navigation buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("Go to Single Customer Prediction"):
        st.switch_page("pages/1_single_customer.py")  # Pastikan path benar

with col2:
    if st.button("Go to Multiple Customer Prediction"):
        st.switch_page("pages/2_multiple_customers.py")

# About Us section
st.markdown("""
    <h3>About Us</h3>
    <div style='text-align: justify;'>
        This Streamlit application is created by the Epsilon team, as the final project for the Job Connector Data Science program at Purwahika Bandung.
        <br><br>
        <b>Epsilon Team Members:</b>
        <ul>
            <li>Eki Nakia Utami</li>
            <li>Naila Firdausi</li>
            <li>Azhar Maulana</li>
        </ul>
    </div>
""", unsafe_allow_html=True)

# Sidebar message
st.sidebar.success('Select a page above to get started.')
