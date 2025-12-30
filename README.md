# House Price Prediction Flask Application

A web application built with Flask that predicts house prices based on various features. The application trains and compares 5 different machine learning models and automatically uses the best performing model.

## Features

- **Model Comparison**: Trains 5 different models (Linear Regression, Random Forest, Decision Tree, Gradient Boosting, SVR)
- **Automatic Best Model Selection**: Automatically selects and saves the best model based on R-squared score
- **Interactive Web UI**: Modern and responsive design for inputting house features
- **Real-time Prediction**: Get instant price predictions with model information

## Models Compared

1. **Linear Regression** - Simple linear regression model
2. **Random Forest Regressor** - Ensemble method using multiple decision trees
3. **Decision Tree Regressor** - Single decision tree model
4. **Gradient Boosting Regressor** - Boosting ensemble method
5. **Support Vector Regressor (SVR)** - Support vector machine for regression

The best model is selected based on the highest R-squared score and is automatically saved for use in predictions.

## Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

## Installation

1. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the machine learning models:**
   ```bash
   python train_model.py
   ```
   
   This will:
   - Load the dataset from `enhanced_house_price_dataset.csv`
   - Preprocess the data (encode categorical variables)
   - Train all 5 models
   - Compare their performance (R-squared, MAE, RMSE)
   - Display a comparison table
   - Automatically save the best performing model
   - Save label encoders and feature names

3. **Run the Flask application:**
   ```bash
   python app.py
   ```

4. **Open your browser and navigate to:**
   ```
   http://localhost:5000
   ```

## Usage

1. Fill in all the house features in the form:
   - Area (in square feet)
   - Number of bedrooms
   - Number of bathrooms
   - Number of stories
   - Number of parking spaces
   - Age of the house (in years)
   - City
   - Furnishing status
   - Main road availability
   - Guest room availability
   - Basement availability
   - Water supply type
   - Air conditioning availability
   - Preferred tenant type
   - Locality rating (1-10)

2. Click the "Predict Price" button

3. The predicted price will be displayed below the form along with the model name used for prediction

## Project Structure

```
house_price_prediction/
├── app.py                          # Flask application with routes
├── train_model.py                  # Model training and comparison script
├── templates/
│   └── index.html                  # HTML template for the UI
├── models/                         # Folder containing all model files
│   ├── house_price_model.pkl      # Best trained model (generated after training)
│   ├── label_encoders.pkl         # Label encoders (generated after training)
│   ├── feature_names.pkl          # Feature names (generated after training)
│   └── model_info.pkl             # Model information (generated after training)
├── enhanced_house_price_dataset.csv # Dataset file
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Model Details

- **Algorithm Selection**: Best of 5 models (Linear Regression, Random Forest, Decision Tree, Gradient Boosting, SVR)
- **Features**: 15 input features (Area, Bedrooms, Bathrooms, Stories, Parking, Age, City, Furnishing, Main Road, Guest Room, Basement, Water Supply, Air Conditioning, Preferred Tenant, Locality Rating)
- **Target**: House Price (in Indian Rupees)
- **Evaluation Metrics**: R-squared Score, Mean Absolute Error (MAE), Root Mean Squared Error (RMSE)

## Notes

- Make sure to train the models (`train_model.py`) before running the Flask app
- All model files (`.pkl` files) are stored in the `models/` folder
- The `models/` folder is automatically created when you run `train_model.py`
- The application runs in debug mode by default for development
- The training script will display a comparison table showing all model performances

## Troubleshooting

- If you get an error about missing model files, run `train_model.py` first
- If port 5000 is already in use, modify the port in `app.py` (last line)
- Make sure all required packages are installed using `pip install -r requirements.txt`
- If SVR training is slow, it's normal - SVR can be computationally intensive
