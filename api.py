# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify
import numpy as np
import joblib

# Load model and preprocessing objects
model = joblib.load('visa_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Initialize Flask app
app = Flask(__name__)

# Function to encode incoming user data
def encode_input(data):
    categorical_cols = [
        # 'Gender', 'Nationality', 'Current_Country', 'Highest_Education', 'Field_of_Study',
        # 'Job_Title', 'Industry', 'Visa_Type', 'Applied_Country',
        # 'Previous_Travel_History', 'Sponsorship_Type'

        'Gender', 'Nationality', 'Current_Country', 'Highest_Education', 'Field_of_Study',
        'Visa_Type', 'Applied_Country', 'Previous_Travel_History', 'Sponsorship_Type'

    ]
    
    for col in categorical_cols:
        le = label_encoders[col]
        if data[col] not in le.classes_:
            data[col] = le.classes_[0]  # Default to first class
        else:
            data[col] = le.transform([data[col]])[0]
    
    return data

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        data_encoded = encode_input(data)
        
        sample = np.array([[
            data_encoded['Age'], 
            data_encoded['Gender'], 
            data_encoded['Nationality'],
            data_encoded['Current_Country'], 
            data_encoded.get('GPA', 0), 
            data_encoded.get('Highest_Education', 0),
            data_encoded.get('Field_of_Study', 0), 
            data_encoded.get('IELTS_Score', 0),
            0,0,0,0,
            # data_encoded.get('Job_Title', 0), 
            # data_encoded.get('Years_Work_Experience', 0),
            # data_encoded.get('Industry', 0), 
            # data_encoded.get('Expected_Salary_USD', 0),
            data_encoded['Visa_Type'], 
            data_encoded['Applied_Country'],
            data_encoded['Previous_Travel_History'],
            0,
            # data_encoded.get('Annual_Income_USD', 0),
            data_encoded['Bank_Statement_Amount'], 
            data_encoded['Sponsorship_Type']
        ]])

        sample_scaled = scaler.transform(sample)
        probability = model.predict_proba(sample_scaled)[0][1]
        
        # Threshold (use the best_threshold you printed in training)
        best_threshold = 0.5  # Replace with your real best_threshold
        prediction = "Approved" if probability > best_threshold else "Rejected"

        return jsonify({
            'prediction': prediction,
            'approval_probability': probability
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
