import numpy as np
import joblib
import os
import pandas as pd

# Load the pre-trained ML model
model_path = os.path.join("models", "best_model.pkl")
preprocessor_path = os.path.join("models", "preprocessor.pkl")

def preprocess_data(x, y):
    # Load the preprocessor
    preprocessor = joblib.load(preprocessor_path)
    
    # Preprocess the input data
    processed_data = preprocessor.transform([[x, y]])
    
    # Create a DataFrame with named columns
    processed_df = pd.DataFrame(processed_data, columns=['x_value', 'y_value'])
    return processed_df


def predict_coordinate(x, y):
    # Preprocess the input data
    processed_data = preprocess_data(x, y)
    
    # Load the pre-trained model
    model = joblib.load(model_path)
    
    # Predict the coordinate using the model
    prediction = model.predict(processed_data)
    return prediction[0]

if __name__ == "__main__":
    # Test data
    x_value = 10.0
    y_value = 20.0
    
    # Predict the coordinate
    prediction = predict_coordinate(x_value, y_value)
    print("Predicted Coordinate:", prediction)
