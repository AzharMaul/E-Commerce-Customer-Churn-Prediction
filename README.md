# E-Commerce-Customer-Churn-Prediction

## 📌 Overview
This project is an **E-Commerce Customer Churn Predictor**, built using **Streamlit** for the web app interface and **XGBoost Classification** for machine learning-based prediction. The goal is to help e-commerce businesses identify customers at risk of churning and take proactive actions to retain them.

## 🔥 Features
- **Customer Churn Prediction:** Uses an XGBoost model to predict whether a customer is likely to churn.
- **RFM Analysis:** Segments customers based on **Recency, Frequency, and Monetary (RFM)** scores.
- **Data Preprocessing:** Automatically normalizes categorical variables (e.g., `PreferredPaymentMode`, `PreferredLoginDevice`).
- **CSV Upload & Download:** Users can upload customer data and download predictions as a CSV file.

## 🧑‍💻 Tech Stack
- **Python**
- **Pandas**
- **XGBoost**
- **Streamlit**
- **Pickle** (for saving and loading models)

## 🛠 Machine Learning Model
### Model Used
- **XGBoost Classifier**: Optimized for handling imbalanced data with a focus on the **F2 Score**, ensuring that high-risk customers are not missed.

### Feature Engineering
The model is trained on multiple customer attributes such as:
- **Transaction Behavior** (Order count, last order date, cashback received)
- **Engagement Patterns** (Preferred login device, preferred payment mode)
- **Demographics** (Location, age, etc.)

### Data Processing Steps
- Categorical variable transformation (e.g., `Phone` → `Mobile Phone`)
- Handling missing values
- RFM scoring & customer segmentation
- Model training using optimized hyperparameters

## 📊 Results & Insights
- Customers are segmented into different categories: **Best Customers, Loyal Customers, Potential Loyalists, At Risk, and Churned Customers**.
- The model helps in early detection of high-risk customers, enabling businesses to implement retention strategies such as **personalized offers and engagement campaigns**.

## 📌 Future Improvements
- Enhance feature selection for better accuracy.
- Integrate real-time customer data streaming.
- Improve model interpretability with SHAP values.

## 📜 License
This project is open-source and available under the MIT License.

---
### 🎯 Contributing
Feel free to fork this repository, create a branch, and submit a pull request. Contributions are always welcome! 😊


