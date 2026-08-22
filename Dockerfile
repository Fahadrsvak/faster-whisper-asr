FROM python:3.10-slim

RUN apt-get update && apt-get install -y ffmpeg libsndfile1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir faster-whisper websockets numpy soundfile

COPY server.py .

EXPOSE 6009
CMD ["python", "server.py"]