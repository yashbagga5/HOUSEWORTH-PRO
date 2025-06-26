Python 3.12.10 (tags/v3.12.10:0cc8128, Apr  8 2025, 12:21:36) [MSC v.1943 64 bit (AMD64)] on win32
Enter "help" below or click "Help" above for more information.

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# Load the pre-trained model
model = joblib.load('RandomForestRegressionModel.joblib')

def predict_price(location, total_sqft, bath, bhk):
    # Create a DataFrame with input features
    input_data = pd.DataFrame({
        'location': [location],
        'total_sqft': [total_sqft],
        'bath': [bath],
        'bhk': [bhk]
    })
    
    # Predict price
    predicted_price = model.predict(input_data)[0]
    return predicted_price

# Main Streamlit app
... st.set_page_config(page_title="Housing Price Predictor", page_icon="🏠")
... st.title("Housing Price Prediction App")
... st.subheader("Predict House Prices in Bengaluru using Machine Learning")
... 
... # Sidebar for user inputs
... st.sidebar.header("User Input Features")
... 
... # Get locations
... location_encoder = model.named_steps['columntransformer'].named_transformers_['onehotencoder']
... locations = [loc.replace('location_', '') for loc in location_encoder.get_feature_names_out(['location'])]
... 
... location = st.sidebar.selectbox("Location", sorted(locations))
... total_sqft = st.sidebar.number_input("Total Square Feet", min_value=100, max_value=10000, value=1000)
... bath = st.sidebar.number_input("Number of Bathrooms", min_value=1, max_value=10, value=1)
... bhk = st.sidebar.number_input("Number of Bedrooms (BHK)", min_value=1, max_value=10, value=2)
... 
... if st.sidebar.button("Predict Price"):
...     predicted_price = predict_price(location, total_sqft, bath, bhk)
...     st.success(f"Predicted House Price: ₹{predicted_price:,.2f} Lakhs")
... 
... if st.sidebar.button("Show Correlation Heatmap"):
...     # Load the cleaned data for heatmap
...     data = pd.read_csv("Cleaned_data.csv")
... 
...     # Select only numerical features
...     numerical_data = data.select_dtypes(include=['number'])
... 
...     # Create heatmap
...     st.subheader("Feature Correlation Heatmap")
...     fig, ax = plt.subplots(figsize=(10, 8))
...     sns.heatmap(numerical_data.corr(), cmap='coolwarm', annot=True, fmt='.2f', linewidths=0.5, ax=ax)
...     st.pyplot(fig)
... 
