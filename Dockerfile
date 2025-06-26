# Use the specified base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file to install dependencies
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install NLTK data for the scorer
RUN python -m nltk.downloader wordnet punkt averaged_perceptron_tagger

# Copy the entire project, including trained models and vectorizer
COPY . .

# Expose port 8000 as specified
EXPOSE 8000

# Set environment variables for Python
ENV PYTHONUNBUFFERED=1

# Command to run the FastAPI app with uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]