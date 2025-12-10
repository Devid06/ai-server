# Use a lightweight Python image
FROM python:3.10-slim

# --- CRITICAL MEMORY FIX ---
# Limit compilation to 1 core. 
# Without this, dlib uses all cores and crashes the build server (OOM).
ENV CMAKE_BUILD_PARALLEL_LEVEL=1

# 1. Install system dependencies required for building dlib
RUN apt-get update && apt-get install -y \
    cmake \
    g++ \
    make \
    git \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. Set working directory
WORKDIR /app

# 3. Copy requirements
COPY requirements.txt .

# 4. Install dependencies
# We use --no-cache-dir to keep the image small
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 5. Manual fix for face_recognition_models (just in case)
RUN pip install --no-cache-dir git+https://github.com/ageitgey/face_recognition_models

# 6. Copy your application code
COPY . .

# 7. Command to run the app
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
