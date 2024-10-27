FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# add app.py and models directory
COPY app.py .
COPY models/ ./models/

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# specify default commands
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]