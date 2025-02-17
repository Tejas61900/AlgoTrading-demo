# Use the official Python image from the Docker Hub
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy requirements file (if you have one)
COPY requirements.txt .

# Install necessary packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your bot code into the container
COPY . .

# Expose a port if your bot runs a web server (optional)
EXPOSE 5000

# Command to run the bot
CMD ["python", "app.py"]