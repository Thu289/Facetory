FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
gcc \
g++ \
libpq-dev \
libgl1 \
libglib2.0-0 \
libsm6 \
libxext6 \
libxrender1 \
libgomp1 \
libfreetype6 \
libpng-dev \
libjpeg-dev \
curl \
&& apt-get clean \
&& rm -rf /var/lib/apt/lists/*


# RUN pip install --no-cache-dir --upgrade pip==25.1.1 setuptools wheel

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install -r requirements.txt

# Copy source code
COPY . .

# Expose port
EXPOSE 8000

# Start the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"] 