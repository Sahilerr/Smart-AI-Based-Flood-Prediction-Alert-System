from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from flask import Flask, render_template, request, jsonify 

app = Flask(__name__)
CORS(app)

# Load classical ML model (pkl)
with open("flood_prediction_model.pkl", "rb") as f:
    classical_model = pickle.load(f)

# Load deep learning model (h5)
deep_model = load_model("flood_detection_model (2).h5")

@app.route('/')
def home():
    return render_template('index.html')

# Route for classical ML model prediction
@app.route("/predict/classical", methods=["POST"])
def predict_classical():
    try:
        data = request.get_json()
        features = np.array(data["features"]).reshape(1, -1)
        prediction = classical_model.predict(features)
        return jsonify({"prediction": int(prediction[0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Route for deep learning model prediction
@app.route("/predict/deeplearning", methods=["POST"])
def predict_deeplearning():
    try:
        data = request.get_json()
        features = np.array(data["features"]).reshape(1, -1)
        prediction = deep_model.predict(features)
        predicted_class = int(np.argmax(prediction))
        return jsonify({"prediction": predicted_class})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
