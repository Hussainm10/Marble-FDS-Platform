FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY gnn_stub/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY gnn_stub/app.py .

# Copy model artifacts
COPY gnn/artifacts/ ./artifacts/

EXPOSE 5001

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5001"]
