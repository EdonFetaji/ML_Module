FROM python:3.10-slim

# Set working directory
WORKDIR /app

ENV PYTHONUNBUFFERED=1

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy project files
COPY . .

# Expose port
EXPOSE 8000

# Run server
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
