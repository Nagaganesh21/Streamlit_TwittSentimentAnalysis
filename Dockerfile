FROM python:3.10

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -m nltk.downloader punkt

# Copy application code
COPY . .

# Expose port
EXPOSE 8080

# Define entrypoint and default command
ENTRYPOINT ["python"]
CMD ["gunicorn", "app:app", "--config=config.py"]
