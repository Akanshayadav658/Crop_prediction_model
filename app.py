import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import os

# Initialize Flask app
app = Flask(__name__)  # 🔁 Renamed from flask_app to app

# Load the trained model
model = pickle.load(open("model.pkl", "rb"))

# Home route
@app.route("/")
def Home():
    return render_template("index4.html")  # Ensure this file exists in a 'templates' folder

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)

    return render_template(
        "index4.html",
        prediction_text="The predicted crop is: {}".format(prediction[0])
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)  # ✅ Now 'app' is defined
