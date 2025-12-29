# Use an official Python runtime as the base image.
FROM python:3.12-slim

# Set environment variables for MLflow tracking server if needed for download
# Replace with your actual MLflow tracking URI
# For example: "http://your-mlflow-tracking-server:5000"
# Or if using Databricks: "databricks" and set MLFLOW_TRACKING_USERNAME/PASSWORD or use Databricks CLI config
# ENV MLFLOW_TRACKING_URI="http://mlflow:5000"
# If your MLflow server requires authentication, you'll need to configure it here, e.g.:
# ENV MLFLOW_TRACKING_USERNAME="your_username"
# ENV MLFLOW_TRACKING_PASSWORD="your_password"
# Or use MLFLOW_TRACKING_TOKEN if applicable.

# Set the working directory inside the container.
WORKDIR /app

# Copy the requirements.txt file into the container.
COPY requirements.txt ./

# Install all Python dependencies listed in requirements.txt.
# --no-cache-dir reduces the image size by not storing build cache.
RUN pip install --no-cache-dir -r requirements.txt

# --- DOWNLOAD MLFLOW MODEL FROM SERVER ---
# Define the MLflow Model URI to download.
# This typically points to a registered model or a specific run artifact.
# Examples:
#   - "models:/your_model_name/Production"  (Registered model, 'Production' stage)
#   - "models:/your_model_name/1"           (Registered model, version 1)
#   - "runs:/<RUN_ID>/<ARTIFACT_PATH>"      (Model from a specific run)
#
# IMPORTANT: Replace "models:/your_model_name/Production" with your actual model URI.
# ARG MLFLOW_MODEL_URI="models:/demo@best"
# Destination directory inside the container for the downloaded model.
# ARG DEST_PATH="/app/mlflow_model"

# Use the mlflow CLI to download the model.
# This requires the MLFLOW_TRACKING_URI to be set correctly above.
# RUN mlflow artifacts download --artifact-uri "${MLFLOW_MODEL_URI}" --dst-path "${DEST_PATH}"

# Install additional libraries based on model requirements.
# RUN pip install --no-cache-dir -r /app/mlflow_model/requirements.txt

# Copy your FastAPI application file into the container.
COPY app.py ./
COPY start.sh ./

# Expose port 8000, so the service is accessible from outside the container.
EXPOSE 8000

# Define the command to run when the container starts.
# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
CMD ["bash", "inference/start.sh"]
