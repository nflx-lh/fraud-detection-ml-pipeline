#!/bin/bash
# Download MLFlow model artifacts
echo "Download model artifacts from MLflow..."

if [ ! -d "${DEST_PATH}" ]; then
  echo "Directory ${DEST_PATH} does not exist. Downloading artifacts..."
  mlflow artifacts download --artifact-uri "${MLFLOW_MODEL_URI}" --dst-path "${DEST_PATH}"
else
  echo "Directory ${DEST_PATH} already exists. Skipping download."
fi

# Start FastAPI server with Uvicorn
echo "Starting Uvicorn..."
exec uvicorn inference.app:app --host 0.0.0.0 --port 8000