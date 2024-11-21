from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
import pickle
import xgboost as xgb

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('loan_default_model.pkl', "rb"))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # UI rendering the results
    # Get values from the form
    features = [
        float(request.form['Debt_To_Income_Ratio']),
        float(request.form['Interest_Rate']),
        int(request.form['Employment_Status']),
        float(request.form['Loan_To_Income_Ratio']),
        float(request.form['Credit_Score']),
        float(request.form['Loan_Amount']),
        float(request.form['Age']),
        int(request.form['Previous_Loans']),
        float(request.form['Payment_To_Income_Ratio']),
        float(request.form['Monthly_Payment'])
    ]

    # Convert values to numpy array
    features = np.array(features).reshape(1, -1)

    # Make prediction
    prediction = model.predict(features)[0]
 
    # Create response
    result = 'Default' if prediction > 0.5 else 'Not Default'

    # Render the predicted result
    return render_template('index.html', 
                            prediction=result)

if __name__ == '__main__':
    app.run(debug=True)