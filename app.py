from flask import Flask, request, jsonify
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

# Load all models and scaler
linear_model = pickle.load(open('linear_model.pkl', 'rb'))
gb_model = pickle.load(open('gb_model.pkl', 'rb'))
knn_model = pickle.load(open('knn_model.pkl', 'rb'))
svr_model = pickle.load(open('svr_model.pkl', 'rb'))
rf_model = pickle.load(open('rf_model.pkl', 'rb'))
scaler = pickle.load(open('Scaler.pkl', 'rb'))


@app.route('/')
def home():
    return "Welcome to the Car Insurance Prediction API!"


@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    data = request.json

    try:
        input_data = [
            data['Driver Age'],
            data['Driver Experience'],
            data['Previous Accidents']
        ]
    except KeyError as e:
        return jsonify({"error": f"Missing key: {str(e)}"}), 400

    # Convert to DataFrame with proper feature names
    feature_names = ['Driver Age', 'Driver Experience', 'Previous Accidents']
    input_df = pd.DataFrame([input_data], columns=feature_names)

    # Scale the data
    scaled_data = scaler.transform(input_df)

    # Get predictions from all models
    predictions = {
        'Linear Regression': float(linear_model.predict(scaled_data)[0]),
        'Gradient Boosting': float(gb_model.predict(scaled_data)[0]),
        'KNN': float(knn_model.predict(scaled_data)[0]),
        'SVR': float(svr_model.predict(scaled_data)[0]),
        'Random Forest': float(rf_model.predict(scaled_data)[0])
    }

    # Calculate average prediction
    average_prediction = np.mean(list(predictions.values()))

    return jsonify({
        # "individual_predictions": predictions
        "ensemble_prediction": average_prediction
        # "message": f"The predicted insurance cost is USD {average_prediction:.2f}"
    })


if __name__ == '__main__':
    app.run(debug=True)