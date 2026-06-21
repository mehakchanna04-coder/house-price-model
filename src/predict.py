import pandas as pd
import pickle
import numpy as np

def load_model(path):
    with open(path, 'rb') as f:
        model = pickle.load(f)
    print("Model loaded!")
    return model

def predict_price(model, input_data):
    prediction = model.predict(input_data)
    return prediction

if __name__ == "__main__":
    # Load model
    model = load_model('outputs/best_model_xgboost.pkl')
    print("Model ready to make predictions!")
    print("Use predict_price(model, your_data) to predict house prices")