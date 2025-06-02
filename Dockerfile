# Python official image
FROM python:3.10-slim

# Working directory
WORKDIR /app

# Copy requirements file and install all depencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project
COPY . .

# Show port 8000 for FastAPI
EXPOSE 8000

# Start command
CMD [ "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000" ]
