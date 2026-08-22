FROM python:3.10-slim

# Install ffmpeg for audio processing capabilities
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

EXPOSE 6009
CMD ["python", "main.py"]