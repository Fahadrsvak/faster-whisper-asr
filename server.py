import asyncio
import json
import logging
import time
import numpy as np
import io
import soundfile as sf
import websockets
from faster_whisper import WhisperModel

logging.basicConfig(level=logging.INFO)

# Switch to 'tiny.en' for high-speed CPU inference on budget VPS instances
MODEL_SIZE = "tiny.en"
logging.info(f"Loading Faster-Whisper model ({MODEL_SIZE})...")
model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8", cpu_threads=2)
logging.info("Faster-Whisper initialized successfully.")

def transcribe_blob(audio_bytes):
    """Processes complete audio blob off-thread using SoundFile decoding."""
    start_time = time.perf_counter()
    
    # Read binary WAV/WebM/OGG buffer into float32 numpy array
    audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))
    
    # Convert stereo to mono if necessary
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)
        
    audio_np = audio_data.astype(np.float32)

    segments, info = model.transcribe(
        audio_np,
        beam_size=1,            # Fast greedy search
        language="en",
        vad_filter=True,        # VAD works reliably on complete recordings
        vad_parameters=dict(min_silence_duration_ms=500)
    )
    
    text = " ".join([segment.text.strip() for segment in segments if segment.text]).strip()
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    
    return text, elapsed_ms, info.duration

async def handle_websocket(websocket):
    logging.info("Client connected")

    try:
        async for message in websocket:
            # Handle incoming full audio blob from client
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