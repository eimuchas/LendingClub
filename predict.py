# from flask import Flask, request, jsonify
# import joblib
# import pandas as pd
# import json
# import os
# from datetime import datetime

# app = Flask(__name__)

# # Load the model

# model_loan_status = joblib.load('lgbm_loan_status.pkl')

# @app.route('/predict/loan_status', methods=['POST'])
# def predict_loan_status():
#     start_time = datetime.now()

#     data = request.json
#     df = pd.DataFrame(data)

#     # Adjusting data types
#     df["loan_amnt"] = df["loan_amnt"].astype(float)
#     df["risk_score"] = df["risk_score"].astype(float)
#     df["debt_to_income"] = df["debt_to_income"].astype(float)
#     df["year_issue"] = df["year_issue"].astype(int)
#     df["fico_range_high"] = df["fico_range_high"].astype(float)
#     df["last_fico_range_low"] = df["last_fico_range_low"].astype(float)
#     df["percent_bc_gt_75"] = df["percent_bc_gt_75"].astype(float)
#     df["loan_amnt"] = df["loan_amnt"].astype(float)
#     df["total_rev_hi_lim"] = df["total_rev_hi_lim"].astype(float)
#     df["inq_last_6mths"] = df["inq_last_6mths"].astype(float)
#     df["acc_open_past_24mths"] = df["acc_open_past_24mths"].astype(float)
#     df["dti"] = df["dti"].astype(float)
#     df["tot_hi_cred_lim"] = df["tot_hi_cred_lim"].astype(float)
#     df['term'] = ' ' + df['term'].str.lstrip()

#     predictions = model_loan_status.predict(df).tolist()

#     # If your model supports it, get the prediction probabilities
#     loan_status_probabilities = model_loan_status.predict_proba(df).tolist()[0]

#     end_time = datetime.now()
#     processing_time = (end_time - start_time).total_seconds()

#     return jsonify({
#         "loan_status": predictions,
#         "loan_status_probability": loan_status_probabilities,
#         "processing_time": processing_time
#     })


# if __name__ == '__main__':
#     port = int(os.environ.get("PORT", 8000))
#     app.run(host='0.0.0.0', port=port)



from fastapi import FastAPI, Request
from pydantic import BaseModel
import joblib
import pandas as pd
from datetime import datetime
from typing import List
import uvicorn
import os

# Initialize FastAPI app
app = FastAPI()

# Load the model
model_path = 'c:/Users/eiman/Desktop/vs_code/Module_3.3/Deployment/lgbm_loan_status.pkl'

# Load the LightGBM model using joblib
model_loan_status = joblib.load(model_path)

# Define request body using Pydantic
class LoanData(BaseModel):
    loan_amnt: float
    risk_score: float
    debt_to_income: float
    year_issue: int
    title: str
    emp_length: float
    year_issue: int

@app.post("/predict/loan_status")
async def predict_loan_status(data: List[LoanData]):
    start_time = datetime.now()
    print(data)

    # Convert request data to DataFrame
    df = pd.DataFrame([d.dict() for d in data])

    # Adjusting data types
    df["loan_amnt"] = df["loan_amnt"].astype(float)
    df["risk_score"] = df["risk_score"].astype(float)
    df["debt_to_income"] = df["debt_to_income"].astype(float)
    df["year_issue"] = df["year_issue"].astype(int)

    # Make predictions
    predictions = model_loan_status.predict(df).tolist()

    # Get prediction probabilities
    loan_status_probabilities = model_loan_status.predict_proba(df).tolist()[0]

    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()

    return {
        "loan_status": predictions,
        "loan_status_probability": loan_status_probabilities,
        "processing_time": processing_time
    }

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host='0.0.0.0', port=port)


# http://localhost:8000/predict/loan_status