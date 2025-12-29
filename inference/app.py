import os
import json
import time
import mlflow
import uvicorn
import pandas as pd
import redis
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from pathlib import Path

# --- Configuration ---
# This path is where the MLflow model artifacts will be downloaded inside the Docker container.
# This must match the DEST_PATH ARGument in your Dockerfile.
MLFLOW_MODEL_PATH = "inference/mlflow_model"

# --- Load MLflow Model ---
try:
    # Get environment variable in python
    # MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    # MLFLOW_MODEL_URI = os.getenv("MLFLOW_MODEL_URI", None)
    # mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)  # Set the MLflow tracking URI
    # model_name = "demo model"  # Specify the name of the model you want to load
    # model_version = "best"
    # model_uri = f"models:/{model_name}@{model_version}"  # Specify the model version
    # model = mlflow.sklearn.load_model(MLFLOW_MODEL_URI)

    model = mlflow.sklearn.load_model(MLFLOW_MODEL_PATH)
    # print(f"MLflow model loaded successfully from {MLFLOW_MODEL_PATH}")
except Exception as e:
    print(f"Error loading MLflow model: {e}")
    model = None

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Fraud Detection Model Inference API",
    description="API for predicting fraudulent transactions using a pre-trained ML model.",
    version="1.0.0"
)

# --- Pydantic Model for Request Body (Input Schema) ---
class CardTransaction(BaseModel):
    transaction_id: str
    card_number: str
    transaction_datetime: str
    amount: float
    use_chip: str
    merchant_state: str
    mcc: str
    errors: str

class PredictionRequest(BaseModel):
    card_transactions: List[CardTransaction]

# --- Health Check Endpoint ---
@app.get("/health")
async def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet or failed to load. Check server logs.")
    return {"status": "ok", "message": "Model is ready for predictions."}

# --- Prediction Endpoint ---
@app.post("/predict")
async def predict(request: PredictionRequest):
    request_data = request.model_dump()
    # print(f"Received prediction request: {request_data}")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Cannot make predictions.")

    try:
        transaction_list = []
        card_db = redis.Redis(host="redis", port=6379, db=0, password=None)
        
        for card_transaction in request_data["card_transactions"]:    
            # Get card data from online inference table
            card_data = card_db.get(card_transaction["card_number"])

            if card_data is not None:
                # Decode the bytes to string and then parse JSON
                card_data = card_data.decode('utf-8')
                card_data = json.loads(card_data)
            else:
                raise HTTPException(status_code=404, detail=f"Card data not found for {card_transaction['card_number']}.")

            # Combine card data with transaction data
            transaction_info = {**card_data, **card_transaction}
            transaction_list.append(transaction_info)

        # Convert the dictionary to a pandas DataFrame
        df = pd.DataFrame(transaction_list)

        ############ IMPORTANT NOTE ############
        # Need to correspond to Gold Layer add_gold_layer_features that uses Spark

        # Convert transaction_datetime to date
        df['date'] = pd.to_datetime(df['transaction_datetime'])            
        df['acct_open_date'] = pd.to_datetime(df['acct_open_date'])
        # Calculate acct_opened_months
        df['acct_opened_months'] = ((df['date'] - df['acct_open_date']).dt.days / 30.0).round(1)
        # Extract year from 'date'
        df['year_of_tr'] = df['date'].dt.year
        # Calculate yrs_since_pin_changed
        df['yrs_since_pin_changed'] = df['year_of_tr'] - df['year_pin_last_changed']
        # Drop 'year_of_tr'
        df = df.drop(columns=['year_of_tr'])

        # Predict with model
        predictions = model.predict(df)

        # Add prediction column to df
        df['is_fraud'] = predictions

        # save the features and predictions
        dir_path = Path("monitoring/inference_data")
        file_path = dir_path/"inference_data.csv"

        dir_path.mkdir(exist_ok=True)

        if not file_path.exists():
            df.to_csv(file_path, index=False)
        else:
            df.to_csv(file_path, mode="a", header=False, index=False)

        # Return a list of dictionaries with transaction_id and is_fraud
        predictions = df[['transaction_id', 'is_fraud']].to_dict(orient='records')
        
        # Log the predictions to redis
        timestamp = int(time.time())    
        inference_db = redis.Redis(host="redis", port=6379, db=1, password=None)
        for pred in predictions:
            pred_bytes = {k.encode('utf-8'): str(v).encode('utf-8') for k, v in pred.items()}
            prediction_key = f"prediction:{timestamp}:{pred['transaction_id']}"

            with inference_db.pipeline() as pipe:
                pipe.hset(prediction_key, mapping=pred_bytes)
                pipe.zadd("predictions_by_time", {prediction_key: timestamp})
                pipe.execute()            
                    
        return predictions
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
