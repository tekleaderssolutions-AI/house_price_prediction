import pandas as pd
import pickle
from flask import Flask, request, render_template
from sklearn.preprocessing import OneHotEncoder, StandardScaler

app = Flask(__name__)

# ------------------ Load Models ------------------ #
MODEL_PATHS = {
    "random_forest": "random_forest_model.pkl",
    "linear_regression": "linear_regression_model.pkl",
    "svm": "svm_model.pkl",
    "decision_tree": "decision_tree_model.pkl"
}

LOADED_MODELS = {name: pickle.load(open(path, "rb")) for name, path in MODEL_PATHS.items()}

# Load pre-fitted OneHotEncoder
with open("ohe.pkl", "rb") as f:
    OHE = pickle.load(f)

# Load StandardScaler
with open("standard_scaler.pkl", "rb") as f:
    SCALER = pickle.load(f)

# Load model columns
with open("model_columns.pkl", "rb") as f:
    MODEL_COLUMNS = pickle.load(f)

# ------------------ Preprocess Input ------------------ #
def preprocess_input(data):
    df = pd.DataFrame([data])

    # Convert numeric fields
    numeric_fields = ["Bedrooms", "Bathrooms", "Age", "Area"]
    for field in numeric_fields:
        df[field] = df[field].astype(float)

    # Create 'rooms' feature
    df["rooms"] = df["Bedrooms"] + df["Bathrooms"]
    df = df.drop(columns=["Bedrooms", "Bathrooms", "Age"])

    # OneHotEncode categorical columns
    cat_cols = ["City", "Water Supply", "Preferred Tenant", "Furnishing"]
    encoded = OHE.transform(df[cat_cols])
    encoded_df = pd.DataFrame(encoded, columns=OHE.get_feature_names_out(cat_cols))

    df = pd.concat([df.drop(columns=cat_cols).reset_index(drop=True),
                    encoded_df.reset_index(drop=True)], axis=1)

    # Convert yes/no columns to 0/1
    yes_no_cols = ["Main Road", "Guest Room", "Basement", "Air Conditioning"]
    df[yes_no_cols] = df[yes_no_cols].replace({"Yes": 1, "No": 0})

    # Ensure correct column order
    df = df.reindex(columns=MODEL_COLUMNS, fill_value=0)

    # Scale all numeric columns
    df[df.columns] = SCALER.transform(df[df.columns])

    return df

# ------------------ Predict ------------------ #
def predict_price(data, model_name):
    if model_name not in LOADED_MODELS:
        raise ValueError("Invalid model selection")
    processed_data = preprocess_input(data)
    model = LOADED_MODELS[model_name]
    prediction = model.predict(processed_data)
    return float(prediction[0])

# ------------------ Routes ------------------ #
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form data
        request_data = request.form.to_dict()

        # Check model selection
        model_name = request_data.pop("model", None)
        if not model_name:
            return render_template("index.html", error="Please select a model.")

        # Predict
        price = predict_price(request_data, model_name)

        # Return result on same page
        return render_template("index.html", prediction=price, model=model_name)

    except Exception as e:
        return render_template("index.html", error=f"Prediction failed: {str(e)}")

# ------------------ Run ------------------ #
if __name__ == "__main__":
    app.run(debug=True)
