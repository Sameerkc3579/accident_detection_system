# Use official lightweight Python image
FROM python:3.10-slim

# Memory Optimization Environment Variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MALLOC_ARENA_MAX=2 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

# Set working directory
WORKDIR /app

# Install system dependencies
# "libgl1" and "libglib2.0-0" are required for OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
# CPU-only PyTorch to save space (Hugging Face Free Tier uses CPU by default)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# --- ⬇️ CRITICAL FIXES FOR HUGGING FACE SPACES ⬇️ ---

# 1. Create necessary folders
RUN mkdir -p static models

# 2. FIX PERMISSIONS: Hugging Face runs as user 1000 (non-root).
# We must give "Write" access (777) to these folders so the app can save videos.
RUN chmod -R 777 static models

# 3. EXPOSE PORT 7860 (Strictly required by Hugging Face)
EXPOSE 7860

# 4. START COMMAND
# Force host to 0.0.0.0 and port to 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]