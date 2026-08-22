import asyncio
import json
import logging
import numpy as np
import websockets
from faster_whisper import WhisperModel

logging.basicConfig(level=logging.INFO)

# Load model with int8 quantization for optimal CPU performance
MODEL_SIZE = "base.en"
logging.info(f"Loading Faster-Whisper model ({MODEL_SIZE})...")
model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8", cpu_threads=4)
logging.info("Faster-Whisper initialized successfully.")

async def handle_websocket(websocket):
    logging.info("Client connected")
    pcm_bytes_buffer = bytearray()
    last_partial_text = ""

    try:
        async for message in websocket:
            # 1. Handle incoming raw PCM16 binary chunks from mic
            if isinstance(message, bytes):
                pcm_bytes_buffer.extend(message)
                
                # Run inference roughly every ~0.5 sec (16,000 bytes = 0.5 sec at 16kHz PCM16)
                if len(pcm_bytes_buffer) >= 16000:
                    # Convert raw PCM16 bytes to float32 array normalized to [-1.0, 1.0]
                    audio_np = np.frombuffer(pcm_bytes_buffer, dtype=np.int16).astype(np.float32) / 32768.0
                    
                    segments, _ = model.transcribe(
                        audio_np,
                        beam_size=1,             # Greedy search for minimal CPU latency
                        language="en",
                        vad_filter=True,         # Silences background noise
                        vad_parameters=dict(min_silence_duration_ms=300)
                    )
                    
                    partial_text = " ".join([segment.text.strip() for segment in segments]).strip()
                    
                    if partial_text and partial_text != last_partial_text:
                        last_partial_text = partial_text
                        await websocket.send(json.dumps({
                            "type": "partial",
                            "text": partial_text
                        }))

            # 2. Handle string signals from client ('EOS')
            elif isinstance(message, str):
                if message == "EOS":
                    logging.info("EOS signal received. Running final transcription pass...")
                    
                    if len(pcm_bytes_buffer) > 0:
                        audio_np = np.frombuffer(pcm_bytes_buffer, dtype=np.int16).astype(np.float32) / 32768.0
                        
                        segments, _ = model.transcribe(
                            audio_np,
                            beam_size=1,
                            language="en",
                            vad_filter=True
                        )
                        
                        final_text = " ".join([segment.text.strip() for segment in segments]).strip()
                    else:
                        final_text = last_partial_text

                    await websocket.send(json.dumps({
                        "type": "final",
                        "text": final_text or last_partial_text
                    }))
                    
                    pcm_bytes_buffer.clear()
                    break

    except websockets.exceptions.ConnectionClosed:
        logging.info("Client disconnected")
    except Exception as e:
        logging.error(f"Error handling request: {e}")

async def main():
    # Bind to port 6009 inside the container/host
    server = await websockets.serve(handle_websocket, "0.0.0.0", 6009)
    logging.info("Faster-Whisper ASR WebSocket server listening on port 6009...")
    await asyncio.get_running_loop().create_future()

if __name__ == "__main__":
    asyncio.run(main())