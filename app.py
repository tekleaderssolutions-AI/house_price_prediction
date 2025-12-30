# Import Flask and related modules for web application
from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd

# Initialize Flask application
app = Flask(__name__)

# Load the trained model and related files from models folder
with open('models/house_price_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

with open('models/feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

# Load model information
try:
    with open('models/model_info.pkl', 'rb') as f:
        model_info = pickle.load(f)
except FileNotFoundError:
    model_info = {'model_name': 'Unknown', 'r2_score': 0}

# Home page route
@app.route('/')
def home():
    """Render the home page with the prediction form."""
    return render_template('index.html', model_name=model_info.get('model_name', 'Unknown'))

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests and return predicted house price."""
    try:
        data = request.form
        
        area = float(data.get('area', 0))
        bedrooms = int(data.get('bedrooms', 0))
        bathrooms = int(data.get('bathrooms', 0))
        stories = int(data.get('stories', 0))
        parking = int(data.get('parking', 0))
        age = int(data.get('age', 0))
        locality_rating = int(data.get('locality_rating', 0))
        
        city = data.get('city', '')
        furnishing = data.get('furnishing', '')
        main_road = data.get('main_road', '')
        guest_room = data.get('guest_room', '')
        basement = data.get('basement', '')
        water_supply = data.get('water_supply', '')
        air_conditioning = data.get('air_conditioning', '')
        preferred_tenant = data.get('preferred_tenant', '')
        
        city_encoded = label_encoders['City'].transform([city])[0]
        furnishing_encoded = label_encoders['Furnishing'].transform([furnishing])[0]
        main_road_encoded = label_encoders['Main Road'].transform([main_road])[0]
        guest_room_encoded = label_encoders['Guest Room'].transform([guest_room])[0]
        basement_encoded = label_encoders['Basement'].transform([basement])[0]
        water_supply_encoded = label_encoders['Water Supply'].transform([water_supply])[0]
        air_conditioning_encoded = label_encoders['Air Conditioning'].transform([air_conditioning])[0]
        preferred_tenant_encoded = label_encoders['Preferred Tenant'].transform([preferred_tenant])[0]
        
        features = np.array([[
            area, bedrooms, bathrooms, stories, parking, age,
            city_encoded, furnishing_encoded, main_road_encoded,
            guest_room_encoded, basement_encoded, water_supply_encoded,
            air_conditioning_encoded, preferred_tenant_encoded, locality_rating
        ]])
        
        prediction = model.predict(features)[0]
        predicted_price = round(prediction, 2)
        
        return jsonify({
            'success': True,
            'predicted_price': predicted_price,
            'formatted_price': f'₹{predicted_price:,.2f}',
            'model_name': model_info.get('model_name', 'Unknown')
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
