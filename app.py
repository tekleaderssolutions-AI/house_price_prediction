# Flask application using OOP classes
from flask import Flask, render_template, request, jsonify
from model_loader import ModelLoader
from prediction_handler import PredictionHandler

# Initialize Flask application
app = Flask(__name__)

# Load models and initialize prediction handler
loader = ModelLoader(models_dir='models')
model_data = loader.load_all()
prediction_handler = PredictionHandler(
    model=model_data['model'],
    label_encoders=model_data['label_encoders']
)

class HousePriceApp:
    """Main application class for house price prediction."""
    
    def __init__(self, app, prediction_handler, model_info):
        """Initialize application with Flask app, prediction handler, and model info."""
        self.app = app
        self.prediction_handler = prediction_handler
        self.model_info = model_info
        self._register_routes()
    
    def _register_routes(self):
        """Register all application routes."""
        @self.app.route('/')
        def home():
            """Render the home page with the prediction form."""
            return render_template(
                'index.html',
                model_name=self.model_info.get('model_name', 'Unknown')
            )
        
        @self.app.route('/predict', methods=['POST'])
        def predict():
            """Handle prediction requests and return predicted house price."""
            try:
                data = request.form
                
                # Extract numerical features
                area = float(data.get('area', 0))
                bedrooms = int(data.get('bedrooms', 0))
                bathrooms = int(data.get('bathrooms', 0))
                stories = int(data.get('stories', 0))
                parking = int(data.get('parking', 0))
                age = int(data.get('age', 0))
                locality_rating = int(data.get('locality_rating', 0))
                
                # Extract categorical features
                city = data.get('city', '')
                furnishing = data.get('furnishing', '')
                main_road = data.get('main_road', '')
                guest_room = data.get('guest_room', '')
                basement = data.get('basement', '')
                water_supply = data.get('water_supply', '')
                air_conditioning = data.get('air_conditioning', '')
                preferred_tenant = data.get('preferred_tenant', '')
                
                # Prepare features and predict
                features = self.prediction_handler.prepare_features(
                    area, bedrooms, bathrooms, stories, parking, age,
                    city, furnishing, main_road, guest_room, basement,
                    water_supply, air_conditioning, preferred_tenant, locality_rating
                )
                
                predicted_price = self.prediction_handler.predict(features)
                formatted_price = self.prediction_handler.format_price(predicted_price)
                
                return jsonify({
                    'success': True,
                    'predicted_price': predicted_price,
                    'formatted_price': formatted_price,
                    'model_name': self.model_info.get('model_name', 'Unknown')
                })
            
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 400

# Initialize application
house_price_app = HousePriceApp(
    app=app,
    prediction_handler=prediction_handler,
    model_info=model_data['model_info']
)

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
