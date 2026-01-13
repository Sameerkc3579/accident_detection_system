# Use official lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
# Using opencv-python-headless is strictly better for server environments to avoid GUI deps
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create static directory
RUN mkdir -p static

# Expose port (Render uses $PORT env var, but defaults to 10000 or similar inside internal network, external mapping handles it)
EXPOSE 8000

# Command to run the application
# Use 0.0.0.0 to listen on all interfaces
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
