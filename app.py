
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import os

# Initialize Flask app
flask_app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))  # Load the trained model

# Home route
@flask_app.route("/")
def Home():
    return render_template("index4.html")  # Ensure index4.html exists in the templates folder

# Prediction route
@flask_app.route("/predict", methods=["POST"])
def predict():
    # Get input data from the form
    float_features = [float(x) for x in request.form.values()]  # Corrected `values()` instead of `value()`
    features = [np.array(float_features)]  # Create a NumPy array for the model
    prediction = model.predict(features)  # Fixed `predictz` to `predict`
    
    # Return prediction to the user
    return render_template(
        "index4.html",
        prediction_text="The predicted crop is: {}".format(prediction[0])  # Access the first element of the prediction
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)


