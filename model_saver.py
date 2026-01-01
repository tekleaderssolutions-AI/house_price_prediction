# Model saving module using OOP
import pickle
import os

class ModelSaver:
    """Class responsible for saving trained models and related files."""
    
    def __init__(self, models_dir='models'):
        """Initialize saver with models directory path."""
        self.models_dir = models_dir
        self._ensure_directory_exists()
    
    def _ensure_directory_exists(self):
        """Create models directory if it doesn't exist."""
        os.makedirs(self.models_dir, exist_ok=True)
    
    def save_model(self, model, filename='house_price_model.pkl'):
        """Save trained model to pickle file."""
        filepath = os.path.join(self.models_dir, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved as '{filepath}'")
        return filepath
    
    def save_label_encoders(self, label_encoders, filename='label_encoders.pkl'):
        """Save label encoders to pickle file."""
        filepath = os.path.join(self.models_dir, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(label_encoders, f)
        print(f"Label encoders saved as '{filepath}'")
        return filepath
    
    def save_feature_names(self, feature_names, filename='feature_names.pkl'):
        """Save feature names to pickle file."""
        filepath = os.path.join(self.models_dir, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(feature_names, f)
        print(f"Feature names saved as '{filepath}'")
        return filepath
    
    def save_model_info(self, model_info, filename='model_info.pkl'):
        """Save model information to pickle file."""
        filepath = os.path.join(self.models_dir, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(model_info, f)
        print(f"Model info saved as '{filepath}'")
        return filepath
    
    def save_all(self, model, label_encoders, feature_names, model_info):
        """Save all model-related files."""
        self.save_model(model)
        self.save_label_encoders(label_encoders)
        self.save_feature_names(feature_names)
        self.save_model_info(model_info)

