import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder, StandardScaler

class Datapreprocessing:
    def __init__(self):
        self.ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.sc = StandardScaler()

    def null_check(self, df):
        return df.isnull().sum()
    
    def dupli(self, df):
        return df.duplicated().sum()
    
    # Feature engineering
    def feature_engineering(self, df, ohe=None):
        # Strip spaces from columns
        df.columns = df.columns.str.strip()

        # Rooms calculation
        if 'Bedrooms' in df.columns and 'Bathrooms' in df.columns:
            df['rooms'] = df['Bedrooms'] + df['Bathrooms']
        df = df.drop(columns=['Bedrooms','Bathrooms','Age'], errors='ignore')

        # One-hot encode categorical features
        cat_cols = ['City', 'Water Supply', 'Preferred Tenant', 'Furnishing']
        oh_data = df[cat_cols]

        if ohe is not None:
            encoded = ohe.transform(oh_data)
        else:
            encoded = self.ohe.fit_transform(oh_data)
            with open('ohe.pkl', 'wb') as f:
                pickle.dump(self.ohe, f)

        encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out() if ohe else self.ohe.get_feature_names_out())
        df = df.drop(columns=cat_cols)
        df = pd.concat([df.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)
        return df

    # Binary encoding
    def binary_encoding(self, df):
        yes_no_cols = ['Main Road', 'Guest Room', 'Basement', 'Air Conditioning']
        for col in yes_no_cols:
            if col in df.columns:
                df[col] = df[col].replace({'Yes': 1, 'No': 0})
        return df

    # Normalization
    def normalization(self, df, scaler=None):
        num_cols = df.select_dtypes(include=['float64','int64']).columns
        if scaler is not None:
            scaled = scaler.transform(df[num_cols])
        else:
            scaled = self.sc.fit_transform(df[num_cols])
            with open('standard_scaler.pkl','wb') as f:
                pickle.dump(self.sc, f)
        df[num_cols] = pd.DataFrame(scaled, columns=num_cols)
        return df
