# Use official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY cv_scoring_pipeline.py .

# Install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Set environment variables placeholder (override at runtime)
ENV OPENAI_API_KEY=your_openai_api_key
ENV PINECONE_API_KEY=your_pinecone_api_key
ENV PINECONE_ENVIRONMENT=your_pinecone_environment
ENV S3_BUCKET_NAME=your_s3_bucket_name
ENV CORRELATION_ID=test-id

# Run script
CMD ["python", "cv_scoring_pipeline.py"]
