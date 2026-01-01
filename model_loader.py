# Model loading module using OOP
import pickle
import os

class ModelLoader:
    """Class responsible for loading trained models and related files."""
    
    def __init__(self, models_dir='models'):
        """Initialize loader with models directory path."""
        self.models_dir = models_dir
        self.model = None
        self.label_encoders = None
        self.feature_names = None
        self.model_info = None
    
    def load_model(self, filename='house_price_model.pkl'):
        """Load trained model from pickle file."""
        filepath = os.path.join(self.models_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        return self.model
    
    def load_label_encoders(self, filename='label_encoders.pkl'):
        """Load label encoders from pickle file."""
        filepath = os.path.join(self.models_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Label encoders file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            self.label_encoders = pickle.load(f)
        return self.label_encoders
    
    def load_feature_names(self, filename='feature_names.pkl'):
        """Load feature names from pickle file."""
        filepath = os.path.join(self.models_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Feature names file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            self.feature_names = pickle.load(f)
        return self.feature_names
    
    def load_model_info(self, filename='model_info.pkl'):
        """Load model information from pickle file."""
        filepath = os.path.join(self.models_dir, filename)
        if not os.path.exists(filepath):
            return {'model_name': 'Unknown', 'r2_score': 0}
        
        with open(filepath, 'rb') as f:
            self.model_info = pickle.load(f)
        return self.model_info
    
    def load_all(self):
        """Load all model-related files."""
        self.load_model()
        self.load_label_encoders()
        self.load_feature_names()
        self.load_model_info()
        return {
            'model': self.model,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'model_info': self.model_info
        }

