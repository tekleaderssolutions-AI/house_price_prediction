import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pickle

df=pd.read_csv('preprocessed_house_price_dataset.csv')

x=df.drop(columns=['Price'])
y=df['Price']



# scaling numerical features
sc=StandardScaler()
df_num=x.select_dtypes(include=['float64', 'int64'])
df_num_scaled=sc.fit_transform(df_num)
df_num_scaled=pd.DataFrame(df_num_scaled,columns=df_num.columns)
scaler_filename = 'standard_scaler.pkl'
with open(scaler_filename, 'wb') as file:
    pickle.dump(sc, file)
    
print(df_num_scaled.head())

df_final=pd.concat([df_num_scaled.reset_index(drop=True),y], axis=1)
df_final.to_csv('final_dataset.csv',index=False)