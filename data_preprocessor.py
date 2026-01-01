# Data preprocessing module using OOP
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class DataPreprocessor:
    """Class responsible for loading and preprocessing house price data."""
    
    def __init__(self, file_path='enhanced_house_price_dataset.csv'):
        """Initialize the preprocessor with dataset file path."""
        self.file_path = file_path
        self.label_encoders = {}
        self.categorical_columns = [
            'City', 'Furnishing', 'Main Road', 'Guest Room',
            'Basement', 'Water Supply', 'Air Conditioning', 'Preferred Tenant'
        ]
        self.df = None
        self.X = None
        self.y = None
        self.X_encoded = None
    
    def load_data(self):
        """Load dataset from CSV file."""
        self.df = pd.read_csv(self.file_path)
        print(f"Dataset shape: {self.df.shape}")
        print(f"Dataset columns: {self.df.columns.tolist()}")
        return self.df
    
    def separate_features_target(self):
        """Separate features and target variable."""
        self.X = self.df.drop('Price', axis=1)
        self.y = self.df['Price']
        return self.X, self.y
    
    def encode_categorical_variables(self):
        """Encode categorical variables using LabelEncoder."""
        self.X_encoded = self.X.copy()
        self.label_encoders = {}
        
        for col in self.categorical_columns:
            le = LabelEncoder()
            self.X_encoded[col] = le.fit_transform(self.X[col])
            self.label_encoders[col] = le
            encoding_map = dict(zip(le.classes_, le.transform(le.classes_)))
            print(f"Encoded {col}: {encoding_map}")
        
        return self.X_encoded, self.label_encoders
    
    def get_feature_names(self):
        """Get list of feature column names."""
        if self.X_encoded is not None:
            return self.X_encoded.columns.tolist()
        return None
    
    def preprocess(self):
        """Complete preprocessing pipeline."""
        self.load_data()
        self.separate_features_target()
        self.encode_categorical_variables()
        return self.X_encoded, self.y, self.label_encoders

