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

# Install system dependencies required for OpenCV
# Install system dependencies
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
# Install CPU-only PyTorch first to keep image small (Render Free Tier doesn't have GPU)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create static directory
RUN mkdir -p static

# Expose port (Render uses $PORT env var, but defaults to 10000 or similar inside internal network, external mapping handles it)
EXPOSE 8000

# Command to run the application
# Use 0.0.0.0 to listen on all interfaces
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
