FROM python:3.13-slim

# Set the working directory
WORKDIR /app

# add app.py and models directory
COPY app.py .
COPY schemas.py .
COPY constants.py .
COPY utils.py .
COPY models/ ./models/
COPY steps/ ./steps/

# Install dependencies (install build tools temporarily in case some packages need compilation)
COPY requirements.txt .
RUN apt-get update \
	&& apt-get install -y --no-install-recommends build-essential gcc g++ libc6-dev \
	&& pip install --no-cache-dir -r requirements.txt \
	&& apt-get remove -y build-essential gcc g++ \
	&& apt-get autoremove -y \
	&& rm -rf /var/lib/apt/lists/*

# specify default commands
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]