# HOUSEWORTH-PRO
 🏠 Bangalore House Price Predictor

A machine learning project to predict house prices in Bangalore based on features like location, square footage, number of bedrooms, and more. Built using **Python**, **Pandas**, **Scikit-Learn**, and **Jupyter Notebook**, this project helps users estimate property prices using regression models and real-world housing data.

---

 📌 Objective

The primary goal of this project is to build a regression model that can accurately predict property prices in Bangalore. The model considers multiple features such as area, location, BHK (bedrooms), and number of bathrooms to provide a reliable price estimate.

---

 🧾 Dataset Description

The dataset was sourced from [Kaggle/real estate listings] and contains the following features:

- **Location** – Area or neighborhood in Bangalore  
- **Size** – Number of bedrooms (e.g., 2 BHK)  
- **Total Sqft** – Total area of the property  
- **Bath** – Number of bathrooms  
- **Price** – Price in lakhs (target variable)  

🧹 **Data Cleaning Steps:**

- Removed outliers based on price per square foot  
- Converted categorical features (e.g., location) into numerical via one-hot encoding  
- Handled missing values and inconsistent entries  
- Normalized features where necessary

---

 🛠 Tools & Technologies

- **Python 3.x**  
- **Jupyter Notebook**  
- **Pandas & NumPy** – Data wrangling  
- **Matplotlib & Seaborn** – Data visualization  
- **Scikit-Learn** – Model building (Linear Regression, Lasso, Ridge)  
- **GridSearchCV** – Hyperparameter tuning  
- **Pickle** – Model serialization for deployment

---

 📈 Features & Workflow

1. **Data Loading & Exploration**  
   - Understand data distribution and correlation using visualizations

2. **Feature Engineering**  
   - Clean and transform `size` and `total_sqft`  
   - Create new columns like `price_per_sqft`  
   - Remove unusual and rare locations

3. **Model Selection**  
   - Tried Linear Regression, Lasso, and Ridge  
   - Compared models using RMSE (Root Mean Squared Error)

4. **Hyperparameter Tuning**  
   - Used GridSearchCV for optimal alpha in Ridge/Lasso

5. **Model Evaluation**  
   - Final model achieved **~85% accuracy** on the test set  
   - RMSE reduced by **15%** after tuning and feature selection

6. **Deployment (Optional)**  
   - Model saved using `pickle`  
   - Can be integrated into a Flask web app or Streamlit interface

---

 📊 Key Insights

- Locations like Indira Nagar, HSR Layout, and Koramangala had higher price per sqft  
- BHK size and area were strongly correlated with price  
- Outlier removal significantly improved model stability

---

 🚀 How to Run

1. Clone the repository:
2. Install dependencies
3. run the notebook


📚 Future Enhancements
Build a Streamlit or Flask web app for predictions

Integrate external APIs for location data

Include more features like property type, furnishing status, etc.

Add map visualizations for spatial trends

👨‍💻 Author
Yash Bagga
📧 yashbagga5@gmail.com
🔗 LinkedIn
🔗 GitHub
