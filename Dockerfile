FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU version first to keep image lightweight
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install FastAPI and Uvicorn serving stack, plus HF transformers
RUN pip install --no-cache-dir fastapi uvicorn transformers

# Copy serving scripts and visual dashboard assets
COPY . /app

EXPOSE 8000

CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
