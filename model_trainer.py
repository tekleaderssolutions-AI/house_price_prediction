# Model training module using OOP
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd

class ModelResult:
    """Class to store model training results."""
    
    def __init__(self, model, name, r2_score, mae, rmse):
        """Initialize model result with metrics."""
        self.model = model
        self.name = name
        self.r2_score = r2_score
        self.mae = mae
        self.rmse = rmse
    
    def to_dict(self):
        """Convert result to dictionary."""
        return {
            'model': self.model,
            'name': self.name,
            'r2_score': self.r2_score,
            'mae': self.mae,
            'rmse': self.rmse
        }

class ModelTrainer:
    """Class responsible for training machine learning models."""
    
    def __init__(self, test_size=0.2, random_state=42):
        """Initialize trainer with split parameters."""
        self.test_size = test_size
        self.random_state = random_state
        self.models = {}
        self.results = []
    
    def initialize_models(self):
        """Initialize all models to be compared."""
        self.models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(
                n_estimators=100, 
                random_state=self.random_state, 
                n_jobs=-1
            ),
            'Decision Tree': DecisionTreeRegressor(random_state=self.random_state),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100, 
                random_state=self.random_state
            ),
            'Support Vector Regressor': SVR(kernel='rbf', C=100, gamma='scale')
        }
        return self.models
    
    def split_data(self, X, y):
        """Split data into training and testing sets."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state
        )
        return X_train, X_test, y_train, y_test
    
    def train_model(self, model, X_train, y_train, model_name):
        """Train a single model."""
        print(f"\nTraining {model_name}...")
        model.fit(X_train, y_train)
        return model
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate model performance and return results."""
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"\n{model_name} Performance:")
        print(f"  R-squared Score: {r2:.4f}")
        print(f"  Mean Absolute Error: ₹{mae:,.2f}")
        print(f"  Root Mean Squared Error: ₹{rmse:,.2f}")
        
        return ModelResult(model, model_name, r2, mae, rmse)
    
    def train_all_models(self, X_train, X_test, y_train, y_test):
        """Train and evaluate all models."""
        self.initialize_models()
        self.results = []
        
        print("\n" + "="*60)
        print("Training and Evaluating Models")
        print("="*60)
        
        for model_name, model in self.models.items():
            trained_model = self.train_model(model, X_train, y_train, model_name)
            result = self.evaluate_model(trained_model, X_test, y_test, model_name)
            self.results.append(result)
        
        return self.results
    
    def get_comparison_dataframe(self):
        """Get comparison dataframe of all model results."""
        comparison_data = {
            'Model': [r.name for r in self.results],
            'R-squared': [r.r2_score for r in self.results],
            'MAE (₹)': [r.mae for r in self.results],
            'RMSE (₹)': [r.rmse for r in self.results]
        }
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('R-squared', ascending=False)
        return df
    
    def get_best_model(self):
        """Get the best performing model based on R-squared score."""
        if not self.results:
            return None
        return max(self.results, key=lambda x: x.r2_score)

