import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df=pd.read_csv('preprocessed_house_price_dataset.csv')

x=df.drop(columns=['Price'])
y=df['Price']



# scaling numerical features
sc=StandardScaler()
df_num=x.select_dtypes(include=['float64', 'int64'])
df_num_scaled=sc.fit_transform(df_num)
df_num_scaled=pd.DataFrame(df_num_scaled,columns=df_num.columns)

print(df_num_scaled.head())

df_final=df_num_scaled
df_final.to_csv('final_dataset.csv',index=False)