import pandas as pd
import pickle
from flask import Flask, request, render_template
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from preprocessing import feature_engineering, binary_encoding, normalization

app = Flask(__name__)

models = {
    "random_forest": "random_forest_model.pkl",
    "linear_regression": "linear_regression_model.pkl",
    "svm": "svm_model.pkl",
    "decision_tree": "decision_tree_model.pkl"
}

load_models = {name: pickle.load(open(path, "rb")) for name, path in models.items()}

with open("ohe.pkl", "rb") as f:
    onehot = pickle.load(f)

with open("standard_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)


with open("model_columns.pkl", "rb") as f:
    columns = pickle.load(f)

def preprocess_input(data):
    df = pd.DataFrame([data])
    df=feature_engineering(df)
    df=binary_encoding(df)
    df=normalization(df)


    df = df.reindex(columns=columns, fill_value=0)


    df[df.columns] = scaler.transform(df[df.columns])

    return df

def predict_price(data, model_name):
    if model_name not in load_models:
        raise ValueError("Invalid model selection")
    processed_data = preprocess_input(data)
    model = load_models[model_name]
    prediction = model.predict(processed_data)
    return float(prediction[0])


@app.route("/", methods=["GET"])
def index():
    return "Welcome to the House Price Prediction API. Go to /index to use the web interface."

@app.route("/index", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    try:   
        request_data = request.form.to_dict()
        model_name = request_data.pop("model", None)
        
        if not model_name:
            return render_template("index.html", error="Please select a model.")
        price = predict_price(request_data, model_name)


        return render_template("index.html", prediction=price, model=model_name)

    except Exception as e:
        return render_template("index.html", error=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
