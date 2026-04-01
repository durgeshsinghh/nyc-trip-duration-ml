FROM python:3.10-slim

WORKDIR /app

# Copy only required files
COPY requirements.txt .
COPY service.py .
COPY models/model.joblib .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8000"]