from flask import Flask, render_template, request
import pickle
import numpy as np
import os

# Initialize Flask application
app = Flask(__name__)

# Load the trained model
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "HDI.pkl")

with open(model_path, "rb") as file:
    model = pickle.load(file)

# Home Page
@app.route("/")
def home():
    return render_template("index.html")

# Prediction Page
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get country name
        country = request.form["country"]

        # Get input values
        hdi_rank = float(request.form["feature1"])
        hdi_2017 = float(request.form["feature2"])
        hdi_2018 = float(request.form["feature3"])
        hdi_2019 = float(request.form["feature4"])

        # Create input array
        features = np.array([[hdi_rank, hdi_2017, hdi_2018, hdi_2019]])

        # Predict HDI (2020)
        prediction = model.predict(features)

        predicted_hdi = round(float(prediction[0]), 4)

        return render_template(
            "result.html",
            prediction=predicted_hdi,
            country=country
        )

    except Exception as e:
        return render_template(
            "result.html",
            prediction="Error: " + str(e),
            country="Unknown"
        )

# Run Flask App
if __name__ == "__main__":
    app.run(debug=True)