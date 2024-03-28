from flask import Flask, render_template, request
import numpy as np
import joblib
from src.prediction.prediction import predict_coordinate

app = Flask(__name__)

# Load the pre-trained ML model
model = joblib.load("models/best_model.pkl")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the input values from the form
        x_value = float(request.form['x_value'])
        y_value = float(request.form['y_value'])
        
        # Predict the coordinate using the ML model
        prediction = predict_coordinate(x_value, y_value)
        
        # Pass the prediction to the template for display
        return render_template('results.html', prediction=prediction)

if __name__ == "__main__":
    app.run(host='localhost', port=8000, debug=True)
