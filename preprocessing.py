import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
import pickle
from sklearn.preprocessing import StandardScaler

df=pd.read_csv('enhanced_house_price_dataset.csv')

print(df.head())
print(df.info())

def null_check(df):
    print(df.isnull().sum())

def dupli(df):
    duplicates = df.duplicated().sum()
    print(duplicates)


num_col=df.select_dtypes(include=['float64', 'int64']).columns
for col in num_col:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
    plt.show()

print(df.columns)

def feature_engineering(df):
    df['rooms']=df['Bedrooms']+df['Bathrooms']
    df=df.drop(columns=['Bedrooms','Bathrooms','Age'])
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    oh_cat_col = df[['City', 'Water Supply', 'Preferred Tenant', 'Furnishing']]
    encoded_data = ohe.fit_transform(oh_cat_col)

    one_hot_filename = 'ohe.pkl'
    with open(one_hot_filename, 'wb') as file:
        pickle.dump(ohe, file)

    

    encoded_df = pd.DataFrame(
        encoded_data,
        columns=ohe.get_feature_names_out()
    )

    
    df_enc_new = df.drop(columns=['City', 'Water Supply', 'Preferred Tenant', 'Furnishing'])
    df_enc_new = pd.concat([df_enc_new.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)


    return df_enc_new


def binary_encoding(df_enc_new):
    yes_no_cols = ['Main Road', 'Guest Room', 'Basement', 'Air Conditioning']
    df_enc_new[yes_no_cols] = df_enc_new[yes_no_cols].replace({'Yes': 1, 'No': 0})
    print(df_enc_new.head())
    return df_enc_new

df_eng = feature_engineering(df)
df_final = binary_encoding(df_eng)

def normalization(x):
    sc=StandardScaler()
    df_num=x.select_dtypes(include=['float64', 'int64'])
    df_num_scaled=sc.fit_transform(df_num)
    df_num_scaled=pd.DataFrame(df_num_scaled,columns=df_num.columns)
    scaler_filename = 'standard_scaler.pkl'
    with open(scaler_filename, 'wb') as file:
        pickle.dump(sc, file)
        
    df_final_preprocessed = pd.concat([df_num_scaled.reset_index(drop=True), y.reset_index(drop=True)], axis=1)

    
    df_final_preprocessed.to_csv('preprocessed_house_price_dataset.csv', index=False)

    return df_final_preprocessed
    
x=df_final.drop(columns=['Price'])
y=df_final['Price']

df_final_preprocessed = normalization(x)

model_columns = df_final_preprocessed.drop(columns=['Price']).columns.tolist()
with open('model_columns.pkl', 'wb') as f:
    pickle.dump(model_columns, f)

print("Preprocessing complete, encoder, scaler, and model columns saved.")