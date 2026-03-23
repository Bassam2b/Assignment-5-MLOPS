# Base image
FROM python:3.10-slim

# Build argument: the MLflow Run ID passed from the CI/CD pipeline 
ARG RUN_ID
ENV RUN_ID=${RUN_ID}

# Model destination inside the container
ENV MODEL_DIR=/opt/ml/model

# Install system build tools and Python dependencies 
COPY requirements.txt .
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    pip install --no-cache-dir -r requirements.txt && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p ${MODEL_DIR} && \
    echo "Downloading model for Run ID: ${RUN_ID} into ${MODEL_DIR}" && \
    echo "${RUN_ID}" > ${MODEL_DIR}/run_id.txt

# Copy application source
WORKDIR /app
COPY . .

# start inference server
CMD ["python", "train.py"]