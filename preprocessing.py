import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder

df=pd.read_csv('enhanced_house_price_dataset.csv')
'''
print(df.head())
print(df.info())

# Check for missing values
print(df.isnull().sum())
#no missing values 

# duplicate rows
duplicates = df.duplicated().sum()
print(duplicates)
#no duplicate rows

# Check for outliers using boxplots
num_col=df.select_dtypes(include=['float64', 'int64']).columns
for col in num_col:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
    plt.show()
'''
# remove unwanted columns
print(df.columns)

df['rooms']=df['Bedrooms']+df['Bathrooms']
df=df.drop(columns=['Bedrooms','Bathrooms','Age'])

# onre hot encoding for categorical columns
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

oh_cat_col = df[['City', 'Water Supply', 'Preferred Tenant', 'Furnishing']]
encoded_data = ohe.fit_transform(oh_cat_col)

encoded_df = pd.DataFrame(
    encoded_data,
    columns=ohe.get_feature_names_out()
)

print(encoded_df.head())

# concatenate the encoded columns back to the original dataframe
df_new=df.drop(columns=['City', 'Water Supply', 'Preferred Tenant', 'Furnishing'])
df_enc_new = pd.concat([df_new.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

print(df_enc_new.head())



yes_no_cols = ['Main Road', 'Guest Room', 'Basement', 'Air Conditioning']

df_enc_new[yes_no_cols] = df_enc_new[yes_no_cols].replace({'Yes': 1, 'No': 0})

print(df_enc_new.head())

df_enc_new.to_csv('preprocessed_house_price_dataset.csv', index=False)