# Use Python 3.8 as base image
FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements_deploy.txt requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project files
COPY src/ src/
COPY outputs/best_model.pth outputs/

# Set Python path to include src directory
ENV PYTHONPATH="${PYTHONPATH}:/app/src"

# Create app.py in the root directory
COPY src/app.py .

# Expose port
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]
