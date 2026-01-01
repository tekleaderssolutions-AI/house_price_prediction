# Prediction handler module using OOP
import numpy as np

class PredictionHandler:
    """Class responsible for handling house price predictions."""
    
    def __init__(self, model, label_encoders):
        """Initialize handler with model and encoders."""
        self.model = model
        self.label_encoders = label_encoders
    
    def encode_categorical_features(self, city, furnishing, main_road, guest_room, 
                                   basement, water_supply, air_conditioning, preferred_tenant):
        """Encode categorical features using label encoders."""
        city_encoded = self.label_encoders['City'].transform([city])[0]
        furnishing_encoded = self.label_encoders['Furnishing'].transform([furnishing])[0]
        main_road_encoded = self.label_encoders['Main Road'].transform([main_road])[0]
        guest_room_encoded = self.label_encoders['Guest Room'].transform([guest_room])[0]
        basement_encoded = self.label_encoders['Basement'].transform([basement])[0]
        water_supply_encoded = self.label_encoders['Water Supply'].transform([water_supply])[0]
        air_conditioning_encoded = self.label_encoders['Air Conditioning'].transform([air_conditioning])[0]
        preferred_tenant_encoded = self.label_encoders['Preferred Tenant'].transform([preferred_tenant])[0]
        
        return {
            'city': city_encoded,
            'furnishing': furnishing_encoded,
            'main_road': main_road_encoded,
            'guest_room': guest_room_encoded,
            'basement': basement_encoded,
            'water_supply': water_supply_encoded,
            'air_conditioning': air_conditioning_encoded,
            'preferred_tenant': preferred_tenant_encoded
        }
    
    def prepare_features(self, area, bedrooms, bathrooms, stories, parking, age,
                        city, furnishing, main_road, guest_room, basement,
                        water_supply, air_conditioning, preferred_tenant, locality_rating):
        """Prepare feature array for prediction."""
        encoded = self.encode_categorical_features(
            city, furnishing, main_road, guest_room, basement,
            water_supply, air_conditioning, preferred_tenant
        )
        
        features = np.array([[
            area, bedrooms, bathrooms, stories, parking, age,
            encoded['city'], encoded['furnishing'], encoded['main_road'],
            encoded['guest_room'], encoded['basement'], encoded['water_supply'],
            encoded['air_conditioning'], encoded['preferred_tenant'], locality_rating
        ]]])
        
        return features
    
    def predict(self, features):
        """Make prediction using the trained model."""
        prediction = self.model.predict(features)[0]
        return round(prediction, 2)
    
    def format_price(self, price):
        """Format price as currency string."""
        return f'₹{price:,.2f}'

