# syntax=docker/dockerfile:1

FROM python:3.9.7-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port
EXPOSE 5000

# Set the command to run when the container starts
CMD ["python", "app.py"]