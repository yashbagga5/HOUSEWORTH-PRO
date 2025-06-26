#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


# In[3]:


data = pd.read_csv('Bengaluru_House_Data.csv')


# In[4]:


data.head()


# In[5]:


print(data.shape)


# In[6]:


data.info()


# In[7]:


#counting unique values in all columns and seperating each column by ****
for column in data.columns:
    print(data[column].value_counts())
    print("*"*20)


# In[8]:


data.isna()


# In[9]:


data.isnull()


# In[10]:


#sum of null values or missing values in the columns
data.isnull().sum()


# In[11]:


#droping some columns as they are of no use in our model
data.drop(columns =['area_type','availability','society','balcony'], inplace=True)


# In[12]:


data.describe()


# In[13]:


#data info after droping some columns
data.info()


# In[14]:


#filling missing values of different columns
print(data['location'].value_counts())


# In[15]:


data['location']= data['location'].fillna('Sarjapur  Road')


# In[16]:


data['size'].value_counts()


# In[17]:


data['size']= data['size'].fillna('2 BHK')


# In[18]:


data['bath'].value_counts()


# In[19]:


data['bath']=data['bath'].fillna(data['bath'].median())


# In[20]:


data.info()


# In[21]:


#splitting size column to get numbers and then converting it(string) to int;to solve bhk and bedroom problem
data['bhk']= data['size'].str.split().str.get(0).astype(int)


# In[22]:


#outliers in the data 
print(data[data.bhk > 20])


# In[23]:


data['total_sqft'].unique()


# In[24]:


#ideal total_sqft value should be a float value
#creating a function to convert numbers within a range('' - '') to float value
def convertRange(x):

    temp = x.split('-')
    if len(temp) == 2:
        return (float(temp[0]) + float(temp[1]))/2
    try:
        return float(x)
    except:
        return None


# In[25]:


#we have converted total_sqft to float values by applying the above made function
data['total_sqft']=data['total_sqft'].apply(convertRange)


# In[26]:


print(data.head())
#bhk is the int value extracted from size
#total_sqft has float values making the data easier to evaluate


# In[27]:


#making a new column- price per square feet
data['price_per_sqft']= data['price']*100000 / data['total_sqft']
#converting price to lacs by multiplying *100000


# In[28]:


print(data['price_per_sqft'])


# In[29]:


print(data.describe())


# In[30]:


print(data['location'].value_counts())


# In[31]:


#It ensures that the location values are cleaned of leading or trailing spaces and then counts the occurrences of each unique location.
data['location'] = data['location'].apply(lambda x: x.strip())
#lambda function that removes any leading or trailing spaces from the string
#.strip(): A Python string method used to remove whitespace or specified characters from both ends of a string.
location_count = data['location'].value_counts()


# In[32]:


print(location_count)
#location count reduced to 1294 from 1305


# In[33]:


#those locations that occur 10 times or fewer in the dataset(1053 in number)
location_count_less_10 = location_count[location_count <=10]
print(location_count_less_10)


# In[34]:


data['location']=data['location'].apply(lambda x: 'other' if x in location_count_less_10 else x)
# replaces less than 10 frequency locations with the label'other'


# In[35]:


print(data['location'].value_counts())
# locations reduced to 242, other are 2885


# **OUTLIER DETECTION AND REMOVAL**

# In[37]:


data.describe()


# In[38]:


#area of one bedroom
(data['total_sqft']/data['bhk']).describe()


# In[39]:


data = data[((data['total_sqft']/data['bhk']) >= 300)]
data.describe()
# data with bedroom size>=300


# In[40]:


print(data.shape)


# In[41]:


data.price_per_sqft.describe()


# In[42]:


#now max price_per_sqft is an outlier
#removing outlier

def remove_outliers_sqft(df):
    df_output = pd.DataFrame()  #initialising an empty dataframe to store filtered data
    for key,subdf in df.groupby('location'):  #For each unique location, a smaller DataFrame (subdf) containing only rows for that location is created.

        m = np.mean(subdf.price_per_sqft)  #Computes the mean (m) of the price_per_sqft column in subdf

        st = np.std(subdf.price_per_sqft)

        gen_df = subdf[(subdf.price_per_sqft > (m-st)) & (subdf.price_per_sqft <= (m+st))]
         #Keeps only rows in subdf where price_per_sqft lies within the range:
            #Lower bound: m - st and Upper bound: m + st 
        df_output = pd.concat([df_output,gen_df], ignore_index =True)
    return df_output
data  = remove_outliers_sqft(data)
data.describe()


# In[43]:


def bhk_outlier_remover(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats= {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft), 
                 'std': np.std(bhk_df.price_per_sqft), 
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'): 
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5: 
               exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values) 
    return df.drop(exclude_indices, axis='index')


# In[44]:


data = bhk_outlier_remover(data)


# In[45]:


print(data.shape)


# In[46]:


print(data)


# In[47]:


data.drop(columns=['size','price_per_sqft'],inplace=True)


# **EDA**

# In[49]:


# Select only numerical features for correlation analysis
numerical_data = data.select_dtypes(include=['number'])

plt.figure(figsize=(12, 6))
sns.heatmap(numerical_data.corr(),
            cmap = 'BrBG',
            fmt = '.2f',
            linewidths = 2,
            annot = True)

#heatmap


# **CLEANED DATA**

# In[51]:


data.head()


# In[52]:


data.to_csv("Cleaned_data.csv")


# In[53]:


X=data.drop(columns=['price'])
y=data['price']


# **Splitting Data**
# 

# In[55]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error


# In[56]:


X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=0)


# In[57]:


print(X_train.shape)
print(X_test.shape)


# In[58]:


column_trans = make_column_transformer((OneHotEncoder(sparse_output =False),['location']), 
                                       remainder='passthrough')


# In[59]:


scaler = StandardScaler()


# **APPLYING LINEAR REGRESSION**

# In[61]:


linear_pipeline = make_pipeline(column_trans,scaler,LinearRegression())


# In[62]:


linear_pipeline.fit(X_train,y_train)


# In[63]:


y_pred_lr = linear_pipeline.predict(X_test)


# In[64]:


#r squared error-It is used to evaluate how well a regression model fits the data.
score_1 = r2_score(y_test, y_pred_lr)
score_2 = mean_absolute_error(y_test, y_pred_lr)

print('R Sqaured Error:', score_1)
print('Mean Absolute Error:', score_2)


# **APPLYING RANDOM FOREST REGRESSION**

# In[66]:


random_forest_pipeline =  make_pipeline(column_trans, scaler, RandomForestRegressor(n_estimators=10, random_state=42))
random_forest_pipeline.fit(X_train, y_train)


# In[67]:


y_pred_rf = random_forest_pipeline.predict(X_test)


# In[68]:


score_11 = r2_score(y_test, y_pred_rf)
score_22 = mean_absolute_error(y_test, y_pred_rf)

print('R Sqaured Error:', score_11)
print('Mean Absolute Error:', score_22)


# **ACTUAL vs PREDICTED**

# In[70]:


plt.scatter(y_test, y_pred_rf)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual Price vs Predicted Price")
plt.show()


# In[71]:


# Visualization of actual vs predicted prices
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred_lr, label="Linear Regression", alpha=0.7)
plt.scatter(y_test, y_pred_rf, label="Random Forest", alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', label="Perfect Prediction")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Comparison of Actual vs Predicted Prices")
plt.legend()
plt.show()


# In[72]:


#saving the model
joblib.dump(random_forest_pipeline, 'RandomForestRegressionModel.joblib')


# In[ ]:




