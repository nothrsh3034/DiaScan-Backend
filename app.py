from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

# Load the trained model and scaler
model = joblib.load("diabetes_classifier_model.pkl")
scaler = joblib.load("scaler_new.pkl") 

# Initialize Flask app
app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=["POST"])
def predict():
    try:
        
        data = request.get_json()
        print(" Received Data:", data)

        # Extract features
        features = np.array([data['Pregnancies'], data['Glucose'], data['BloodPressure'],
                             data['SkinThickness'], data['Insulin'], data['BMI'],
                             data['DiabetesPedigreeFunction'], data['Age']]).reshape(1, -1)

        print(" Model Input Features (Before Scaling):", features)

        # Standardize the input
        std_features = scaler.transform(features)  
        print(" Standardized Features:", std_features)

        # Make prediction
        prediction = model.predict(std_features)
        print(" Model Prediction:", prediction)

        return jsonify({"prediction": int(prediction[0])})  

    except Exception as e:
        print(" Error:", str(e))
        return jsonify({"error": str(e)})

@app.route("/")
def home():
    return "Flask API is running!"

# Run Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
