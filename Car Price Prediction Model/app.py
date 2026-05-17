from flask import Flask, render_template, request
import numpy as np
import pickle
import joblib

app = Flask(__name__)

# Load model and encoders
model = pickle.load(open(r'C:\Users\zbook\Desktop\Task 14(Project)\model_rf.pkl', 'rb'))
encoders = joblib.load(r'C:\Users\zbook\Desktop\Task 14(Project)\encoders.pkl')

def encode_cols(le, value):
    if value in le.classes_:
        return le.transform([value])[0]
    else:
        return -1

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect form inputs
    user_input = {
        "name": request.form['name'],
        "year": int(request.form['year']),
        "km_driven": int(request.form['km_driven']),
        "fuel": request.form['fuel'],
        "seller_type": request.form['seller_type'],
        "transmission": request.form['transmission'],
        "owner": request.form['owner'],
        "seats": int(request.form['seats']),
        "max_power (in bph)": float(request.form['max_power']),
        "Mileage Unit": request.form['mileage_unit'],
        "Mileage": float(request.form['mileage']),
        "Engine (CC)": int(request.form['engine_cc'])
    }

    # Encode categorical columns
    for col in ['name', 'fuel', 'seller_type', 'transmission', 'owner', 'Mileage Unit']:
        user_input[col] = encode_cols(encoders[col], user_input[col])

    features = [
        'name', 'year', 'km_driven', 'fuel', 'seller_type',
        'transmission', 'owner', 'seats', 'max_power (in bph)',
        'Mileage Unit', 'Mileage', 'Engine (CC)'
    ]

    X = np.array([[user_input[cols] for cols in features]])
    prediction = model.predict(X)[0]
    prediction_real = prediction * 1000
    return render_template('index.html', prediction_text=f"🚗 Predicted Price: {prediction:.2f}")

if __name__ == "__main__":
    app.run(debug=True)

