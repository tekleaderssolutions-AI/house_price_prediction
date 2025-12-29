import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Import dataset
data = pd.read_csv(r"C:\Users\Srikanth Tata\Downloads\enhanced_house_price_dataset.csv")
# print(data.head())

# Feature Engineering and EDA

# print(data.isnull().sum())  # zero null values
# print(data.duplicated().sum()) # zero duplicate values
# print(data.info())
# print(data.describe())

# onehot encoding for the categorical varialbels
categorical_data = data.select_dtypes(include=['object'])
data_encoded = pd.get_dummies(data, columns=categorical_data.columns, drop_first=True, dtype=np.int8)
# print(data_encoded.head())

# Select numeric coloumns only
numeric_data =  data.select_dtypes(include=["int64", "float64"])

# Data Visualization
# Correlation Heatmap
plt.figure(figsize=(10, 10))
sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm',)
plt.title('Correlation Heatmap')
plt.show()

# Sacter Plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x="Area", y="Price", data=numeric_data)
plt.title("Area vs Price Scatter Plot")
plt.xlabel("Area")
plt.ylabel("Price")
plt.show()

# Histogram
plt.figure(figsize=(8, 6))
sns.histplot(numeric_data['Price'], bins=30, kde=True)
plt.title("Price Distribution")
plt.xlabel("price")
plt.ylabel("Freqency")
plt.show()

# Download the cleaned and encoded dataset
data_encoded.to_csv(r"C:\Users\Srikanth Tata\Downloads\cleaned_encoded_house_price_dataset.csv", index=False)