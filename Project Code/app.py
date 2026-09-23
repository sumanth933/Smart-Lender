"""
Smart Lender – Flask Web Application
======================================
Serves two routes:
  GET  /         →  Home page with the loan application form
  POST /predict  →  Processes form data and returns the prediction result

Run with:
    python app.py
"""

from flask import Flask, render_template, request
import numpy as np
import pickle

# ── Initialise Flask app ──────────────────────────────────────────────────────
app = Flask(__name__)

# ── Load the pre-trained XGBoost model ───────────────────────────────────────
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, 'model.pkl'), 'rb') as f:
    model = pickle.load(f)

# ── Label encoding maps (must match train_model.py encoding order) ────────────
GENDER_MAP       = {'Male': 1, 'Female': 0}
MARRIED_MAP      = {'Yes': 1, 'No': 0}
DEPENDENTS_MAP   = {'0': 0, '1': 1, '2': 2, '3+': 3}
EDUCATION_MAP    = {'Graduate': 0, 'Not Graduate': 1}
SELF_EMP_MAP     = {'Yes': 1, 'No': 0}
PROPERTY_MAP     = {'Rural': 0, 'Semiurban': 1, 'Urban': 2}


@app.route('/')
def home():
    """Render the home / loan application form page."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Collect form values, encode them, run the model, and show the result.
    Expected form fields:
        gender, married, dependents, education, self_employed,
        applicant_income, coapplicant_income, loan_amount,
        loan_amount_term, credit_history, property_area
    """
    try:
        # ── Read and encode form inputs ───────────────────────────────────────
        gender            = GENDER_MAP[request.form['gender']]
        married           = MARRIED_MAP[request.form['married']]
        dependents        = DEPENDENTS_MAP[request.form['dependents']]
        education         = EDUCATION_MAP[request.form['education']]
        self_employed     = SELF_EMP_MAP[request.form['self_employed']]
        applicant_income  = float(request.form['applicant_income'])
        coapplicant_income = float(request.form['coapplicant_income'])
        loan_amount       = float(request.form['loan_amount'])
        loan_amount_term  = float(request.form['loan_amount_term'])
        credit_history    = float(request.form['credit_history'])
        property_area     = PROPERTY_MAP[request.form['property_area']]

        # ── Build feature array in the same column order as training ──────────
        features = np.array([[
            gender, married, dependents, education, self_employed,
            applicant_income, coapplicant_income, loan_amount,
            loan_amount_term, credit_history, property_area
        ]])

        # ── Run prediction ────────────────────────────────────────────────────
        prediction = model.predict(features)[0]

        # Model encodes: 0 = Rejected (N), 1 = Approved (Y)
        if prediction == 1:
            result  = 'Approved'
            message = 'Congratulations! Your loan application has been approved.'
            css_class = 'approved'
        else:
            result  = 'Rejected'
            message = 'Unfortunately, your loan application has been rejected.'
            css_class = 'rejected'

        # ── Collect applicant summary to display on result page ───────────────
        applicant = {
            'Gender'            : request.form['gender'],
            'Married'           : request.form['married'],
            'Dependents'        : request.form['dependents'],
            'Education'         : request.form['education'],
            'Self Employed'     : request.form['self_employed'],
            'Applicant Income'  : f"₹ {applicant_income:,.0f}",
            'Co-applicant Income': f"₹ {coapplicant_income:,.0f}",
            'Loan Amount'       : f"₹ {loan_amount * 1000:,.0f}",
            'Loan Term'         : f"{int(loan_amount_term)} months",
            'Credit History'    : 'Good' if credit_history == 1.0 else 'Bad',
            'Property Area'     : request.form['property_area'],
        }

        return render_template(
            'result.html',
            result=result,
            message=message,
            css_class=css_class,
            applicant=applicant
        )

    except Exception as e:
        # Return a friendly error page if something goes wrong
        return render_template(
            'result.html',
            result='Error',
            message=f'An error occurred while processing your request: {str(e)}',
            css_class='rejected',
            applicant={}
        )


if __name__ == '__main__':
    # Debug mode is fine for local development; disable for production
    app.run(debug=True, host='0.0.0.0', port=5000)
