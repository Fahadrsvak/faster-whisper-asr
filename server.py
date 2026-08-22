import asyncio
import json
import logging
import time
import subprocess
import numpy as np
import websockets
from faster_whisper import WhisperModel

logging.basicConfig(level=logging.INFO)

MODEL_SIZE = "tiny.en"
logging.info(f"Loading Faster-Whisper model ({MODEL_SIZE})...")
model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8", cpu_threads=2)
logging.info("Faster-Whisper initialized successfully.")

def decode_audio_with_ffmpeg(audio_bytes):
    """Converts WebM/OGG/WAV audio bytes into 16kHz mono float32 array using FFmpeg."""
    cmd = [
        "ffmpeg",
        "-i", "pipe:0",           # Read binary from stdin
        "-f", "s16le",            # Convert to raw 16-bit PCM LE
        "-ac", "1",               # Convert to 1 channel (Mono)
        "-ar", "16000",           # Resample to 16,000 Hz
        "pipe:1"                  # Output raw PCM to stdout
    ]
    
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    pcm_out, err = process.communicate(input=audio_bytes)
    
    if process.returncode != 0:
        logging.error(f"FFmpeg decoding error: {err.decode('utf-8')}")
        return np.array([], dtype=np.float32)
        
    # Convert raw PCM16 bytes to float32 normalized [-1.0, 1.0]
    return np.frombuffer(pcm_out, dtype=np.int16).astype(np.float32) / 32768.0

def transcribe_blob(audio_bytes):
    """Processes complete decoded audio off-thread."""
    start_time = time.perf_counter()
    
    audio_np = decode_audio_with_ffmpeg(audio_bytes)
    
    if len(audio_np) == 0:
        return "", 0, 0.0

    segments, info = model.transcribe(
        audio_np,
        beam_size=1,
        language="en",
        vad_filter=False
    )
    
    text = " ".join([segment.text.strip() for segment in segments if segment.text]).strip()
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    
    return text, elapsed_ms, info.duration

async def handle_websocket(websocket):
    logging.info("Client connected")

    try:
        async for message in websocket:
            if isinstance(message, bytes):
                logging.info(f"Received audio blob ({len(message)} bytes). Transcribing...")
                
                final_text, inference_ms, audio_dur = await asyncio.to_thread(transcribe_blob, message)
                
                logging.info(
                    f"[BATCH RESULT] Audio: {audio_dur:.2f}s | "
                    f"Inference: {inference_ms:.1f}ms | "
                    f"Text: '{final_text}'"
                )
                
                await websocket.send(json.dumps({
                    "type": "final",
                    "text": final_text
                }))

    except websockets.exceptions.ConnectionClosed:
        logging.info("Client disconnected")
    except Exception as e:
        logging.error(f"Error handling request: {e}")

async def main():
    server = await websockets.serve(handle_websocket, "0.0.0.0", 6009)
    logging.info("Faster-Whisper ASR server listening on port 6009...")
    await asyncio.get_running_loop().create_future()

if __name__ == "__main__":
    asyncio.run(main())