# Import necessary libraries for data processing and machine learning
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import os

# Load and preprocess the dataset
def load_and_preprocess_data():
    """Load the dataset and preprocess categorical variables."""
    df = pd.read_csv('enhanced_house_price_dataset.csv')
    print("Dataset shape:", df.shape)
    print("Dataset columns:", df.columns.tolist())
    
    X = df.drop('Price', axis=1)
    y = df['Price']
    
    X_encoded = X.copy()
    label_encoders = {}
    categorical_columns = ['City', 'Furnishing', 'Main Road', 'Guest Room', 
                         'Basement', 'Water Supply', 'Air Conditioning', 'Preferred Tenant']
    
    for col in categorical_columns:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X[col])
        label_encoders[col] = le
        print(f"Encoded {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    return X_encoded, y, label_encoders

# Initialize and train multiple models
def initialize_models():
    """Initialize all models to be compared."""
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'Support Vector Regressor': SVR(kernel='rbf', C=100, gamma='scale')
    }
    return models

# Evaluate model performance
def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model and return performance metrics."""
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"\n{model_name} Performance:")
    print(f"  R-squared Score: {r2:.4f}")
    print(f"  Mean Absolute Error: ₹{mae:,.2f}")
    print(f"  Root Mean Squared Error: ₹{rmse:,.2f}")
    
    return {
        'model': model,
        'name': model_name,
        'r2_score': r2,
        'mae': mae,
        'rmse': rmse
    }

# Main training function
def main():
    """Main function to train, compare, and save the best model."""
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    X_encoded, y, label_encoders = load_and_preprocess_data()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42
    )
    
    models = initialize_models()
    results = []
    
    print("\n" + "="*60)
    print("Training and Evaluating Models")
    print("="*60)
    
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        model.fit(X_train, y_train)
        result = evaluate_model(model, X_test, y_test, model_name)
        results.append(result)
    
    print("\n" + "="*60)
    print("Model Comparison Summary")
    print("="*60)
    
    comparison_df = pd.DataFrame({
        'Model': [r['name'] for r in results],
        'R-squared': [r['r2_score'] for r in results],
        'MAE (₹)': [r['mae'] for r in results],
        'RMSE (₹)': [r['rmse'] for r in results]
    })
    
    comparison_df = comparison_df.sort_values('R-squared', ascending=False)
    print("\n", comparison_df.to_string(index=False))
    
    best_result = max(results, key=lambda x: x['r2_score'])
    print(f"\n{'='*60}")
    print(f"Best Model: {best_result['name']}")
    print(f"R-squared Score: {best_result['r2_score']:.4f}")
    print(f"Mean Absolute Error: ₹{best_result['mae']:,.2f}")
    print(f"{'='*60}")
    
    # Save all pickle files to models folder
    with open('models/house_price_model.pkl', 'wb') as f:
        pickle.dump(best_result['model'], f)
    print(f"\nBest model ({best_result['name']}) saved as 'models/house_price_model.pkl'")
    
    with open('models/label_encoders.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)
    print("Label encoders saved as 'models/label_encoders.pkl'")
    
    feature_names = X_encoded.columns.tolist()
    with open('models/feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)
    print("Feature names saved as 'models/feature_names.pkl'")
    
    with open('models/model_info.pkl', 'wb') as f:
        pickle.dump({'model_name': best_result['name'], 'r2_score': best_result['r2_score']}, f)
    print("Model info saved as 'models/model_info.pkl'")
    
    print("\nModel training completed successfully!")

if __name__ == '__main__':
    main()
