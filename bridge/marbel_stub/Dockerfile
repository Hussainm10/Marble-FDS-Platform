FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY marbel_stub/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY marbel_stub/app.py .

# Copy model artifacts
COPY marbel/artifacts/ ./artifacts/

EXPOSE 5000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]
