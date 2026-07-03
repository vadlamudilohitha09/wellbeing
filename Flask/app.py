from flask import Flask, render_template, request
import pickle
import numpy as np

# Initialize Flask application
app = Flask(__name__)

# Load the trained model
with open("HDI.pkl", "rb") as file:
    model = pickle.load(file)

# Home page
@app.route('/')
def home():
    return render_template("index.html")

# Prediction page
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Read input values from the form
        features = [float(x) for x in request.form.values()]

        # Convert to NumPy array
        final_features = np.array([features])

        # Predict
        prediction = model.predict(final_features)

        # Round prediction
        output = round(float(prediction[0]), 4)

        return render_template(
            "result.html",
            prediction=output
        )

    except Exception as e:
        return render_template(
            "result.html",
            prediction=f"Error: {e}"
        )

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)