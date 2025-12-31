import pandas as pd
import pickle
import numpy as np
from flask import Flask, request, render_template
from preprocessing import Datapreprocessing

class HousePriceApp:
    def __init__(self):
        self.app = Flask(__name__)
        self.models = {
            "linear_regression": "linear_regression_model.pkl",
            "random_forest": "random_forest_model.pkl",
            "decision_tree": "decision_tree_model.pkl",
            "svm": "svm_model.pkl"
        }

        self.load_models()
        self.load_preprocessors()
        self.register_routes()

    def load_models(self):
        self.loaded_models = {name: pickle.load(open(path, "rb")) for name, path in self.models.items()}

    def load_preprocessors(self):
        with open("ohe.pkl","rb") as f:
            self.onehot = pickle.load(f)
        with open("standard_scaler.pkl","rb") as f:
            self.scaler = pickle.load(f)
        with open("model_columns.pkl","rb") as f:
            self.columns = pickle.load(f)

    def preprocess_input(self, data):
        # Convert input dictionary to DataFrame
        df = pd.DataFrame([data])

        # Convert numeric columns to numeric type
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except:
                pass

        preprocessing = Datapreprocessing()

        # Create 'rooms' feature
        if 'Bedrooms' in df.columns and 'Bathrooms' in df.columns:
            df['rooms'] = df['Bedrooms'] + df['Bathrooms']

        # Binary encode Yes/No columns
        df = preprocessing.binary_encoding(df)

        # One-hot encode categorical columns using saved OHE
        cat_cols = ['City','Water Supply','Preferred Tenant','Furnishing']
        encoded = self.onehot.transform(df[cat_cols])
        encoded_df = pd.DataFrame(encoded, columns=self.onehot.get_feature_names_out(cat_cols))

        # Drop original categorical columns and concatenate encoded
        df = df.drop(columns=cat_cols)
        df = pd.concat([df.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

        # Select numeric + binary columns and encoded categorical columns
        numeric_cols = ['Area','Stories','Parking','Locality Rating','rooms']
        binary_cols = ['Main Road','Guest Room','Basement','Air Conditioning']
        df = df[numeric_cols + binary_cols + list(encoded_df.columns)]

        # Align columns with training columns
        df = df.reindex(columns=self.columns, fill_value=0)

        # Scale all columns
        df[df.columns] = self.scaler.transform(df[df.columns])

        return df



    def predict_price(self, data, model_name):
        if model_name not in self.loaded_models:
            raise ValueError("Invalid model selection")

        df = self.preprocess_input(data)
        model = self.loaded_models[model_name]
        pred = model.predict(df)

        # Log-inverse for linear regression
        if model_name == "linear_regression":
            pred = np.expm1(pred)

        return float(pred[0])

    def register_routes(self):
        @self.app.route("/", methods=["GET"])
        def index():
            return "Welcome! Go to /index for the web form."

        @self.app.route("/index", methods=["GET"])
        def home():
            return render_template("index.html")

        @self.app.route("/predict", methods=["POST"])
        def predict():
            try:
                data = request.form.to_dict()
                model_name = data.pop("model", None)
                if not model_name:
                    return render_template("index.html", error="Select a model.")
                price = self.predict_price(data, model_name)
                return render_template("index.html", prediction=price, model=model_name)
            except Exception as e:
                return render_template("index.html", error=f"Prediction failed: {str(e)}")

    def run(self):
        self.app.run(debug=True)

if __name__ == "__main__":
    app = HousePriceApp()
    app.run()
