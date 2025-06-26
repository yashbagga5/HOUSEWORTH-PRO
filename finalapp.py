#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Set page configuration at the beginning
st.set_page_config(page_title="Housing Price Predictor", page_icon="🏠", layout="wide")

# Title and description
st.title("Bengaluru House Price Prediction App")
st.markdown("### Predict House Prices in Bengaluru using Machine Learning")

# In[2]:


# Load and preprocess the data
@st.cache_data
def load_and_preprocess_data():
    try:
        # Read the dataset
        data = pd.read_csv('Bangalore  house data.csv')
        
        # Display initial info
        st.sidebar.write(f"Dataset loaded with {data.shape[0]} rows and {data.shape[1]} columns")
        
        # Drop unnecessary columns
        data = data.drop(['area_type', 'availability', 'society', 'balcony'], axis=1)
        
        # Handle missing values
        data['location'] = data['location'].fillna('Sarjapur Road')
        data['size'] = data['size'].fillna('2 BHK')
        data['bath'] = data['bath'].fillna(data['bath'].median())
        
        # Extract BHK from size
        data['bhk'] = data['size'].str.split().str.get(0).astype(str)
        data['bhk'] = data['bhk'].apply(lambda x: x if x.isdigit() else '2')
        data['bhk'] = data['bhk'].astype(int)
        
        # Convert total_sqft to float
        def convert_range(x):
            if isinstance(x, str):
                temp = x.split('-')
                if len(temp) == 2:
                    return (float(temp[0]) + float(temp[1]))/2
                try:
                    return float(x)
                except:
                    return 1000  # Default value if conversion fails
            return x
        
        data['total_sqft'] = data['total_sqft'].apply(convert_range)
        
        # Clean location names
        data['location'] = data['location'].apply(lambda x: str(x).strip() if isinstance(x, str) else 'unknown')
        
        # Group less frequent locations
        location_counts = data['location'].value_counts()
        locations_less_than_10 = location_counts[location_counts <= 10].index
        data['location'] = data['location'].apply(lambda x: 'Other' if x in locations_less_than_10 else x)
        
        # Remove outliers
        data = data[data['total_sqft'] > 300]  # Removing extremely small properties
        data = data[data['total_sqft'] < 10000]  # Removing extremely large properties
        data = data[data['bath'] < 10]  # Reasonable number of bathrooms
        data = data[data['price'] < 500]  # Reasonable price cap (in lakhs)
        data = data[data['price'] > 10]  # Minimum reasonable price
        
        # Remove rows with NaN
        data = data.dropna()
        
        return data
    except Exception as e:
        st.error(f"Error loading or preprocessing data: {str(e)}")
        return None

# Create and train the model
@st.cache_resource
def create_model(data):
    try:
        if data is None or len(data) == 0:
            st.error("No valid data available to train the model.")
            return None, ['No locations available']
        
        # Create preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('location', OneHotEncoder(handle_unknown='ignore'), ['location'])
            ],
            remainder='passthrough'
        )
        
        # Create model pipeline
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        
        # Prepare features and target
        X = data[['location', 'total_sqft', 'bath', 'bhk']]
        y = data['price']
        
        # Train the model
        model.fit(X, y)
        
        return model, sorted(data['location'].unique())
    except Exception as e:
        st.error(f"Error creating or training model: {str(e)}")
        return None, ['No locations available']

# Main application
def main():
    # Create layout with two columns
    col1, col2 = st.columns([1, 2])

    with col1:
        st.sidebar.header("User Input Features")
        
        # Load and preprocess data
        with st.spinner('Loading and preprocessing data...'):
            data = load_and_preprocess_data()
        
        if data is None:
            st.error("Could not load or preprocess the data. Please check the dataset.")
            return
        
        # Create the model
        with st.spinner('Training the model...'):
            model, locations = create_model(data)
        
        if model is None:
            st.error("Could not create or train the model. Please check the dataset.")
            return
        
        # Function to predict price
        def predict_price(location, total_sqft, bath, bhk):
            try:
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
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
                return None
        
        # User inputs
        location = st.sidebar.selectbox("Location", locations)
        total_sqft = st.sidebar.slider("Total Square Feet", min_value=300, max_value=10000, value=1000, step=100)
        bath = st.sidebar.slider("Number of Bathrooms", min_value=1, max_value=9, value=2)
        bhk = st.sidebar.slider("Number of Bedrooms (BHK)", min_value=1, max_value=9, value=2)
        
        # Prediction button
        if st.sidebar.button("Predict Price", type="primary"):
            predicted_price = predict_price(location, total_sqft, bath, bhk)
            if predicted_price is not None:
                st.sidebar.success(f"### Predicted House Price: ₹{predicted_price:,.2f} Lakhs")
                
                # Display a gauge chart for the price
                with col2:
                    # Create a progress bar to visualize price range
                    st.subheader("Price Range Indicator")
                    price_range = 500  # Maximum price in lakhs
                    percentage = min(predicted_price / price_range, 1.0)
                    st.progress(percentage)
                    
                    # Price brackets
                    if predicted_price < 50:
                        st.info("🏠 Budget Friendly Range")
                    elif predicted_price < 100:
                        st.info("🏠 Mid Range")
                    elif predicted_price < 200:
                        st.warning("🏠 Premium Range")
                    else:
                        st.error("🏠 Luxury Range")
    
    # Data analysis tabs
    with col2:
        st.subheader("Data Analysis")
        tab1, tab2, tab3 = st.tabs(["Correlation Heatmap", "Price vs Area", "Location Analysis"])
        
        with tab1:
            # Create correlation heatmap
            try:
                numerical_data = data.select_dtypes(include=['number'])
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(numerical_data.corr(), cmap='coolwarm', annot=True, fmt='.2f', linewidths=0.5, ax=ax)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error creating heatmap: {str(e)}")
        
        with tab2:
            # Create scatter plot
            try:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.scatterplot(data=data, x='total_sqft', y='price', hue='bhk', palette='viridis', alpha=0.5)
                plt.xlabel('Total Square Feet')
                plt.ylabel('Price (Lakhs)')
                plt.title('Price vs Area by BHK')
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error creating scatter plot: {str(e)}")
        
        with tab3:
            # Show location-wise average prices
            try:
                location_avg = data.groupby('location')['price'].mean().sort_values(ascending=False).head(10)
                fig, ax = plt.subplots(figsize=(12, 6))
                location_avg.plot(kind='bar', ax=ax, color='teal')
                plt.xticks(rotation=45, ha='right')
                plt.xlabel('Location')
                plt.ylabel('Average Price (Lakhs)')
                plt.title('Top 10 Locations by Average Price')
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error creating location analysis: {str(e)}")

if __name__ == "__main__":
    main()

