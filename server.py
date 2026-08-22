import asyncio
import json
import logging
import numpy as np
import websockets
from faster_whisper import WhisperModel

logging.basicConfig(level=logging.INFO)

MODEL_SIZE = "base.en"
logging.info(f"Loading Faster-Whisper model ({MODEL_SIZE})...")
model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8", cpu_threads=4)
logging.info("Faster-Whisper initialized successfully.")

def run_transcription(audio_np, disable_vad=False):
    """Off-thread Whisper inference."""
    segments, _ = model.transcribe(
        audio_np,
        beam_size=1,
        language="en",
        vad_filter=not disable_vad,
        vad_parameters=dict(threshold=0.2, min_silence_duration_ms=500) if not disable_vad else None
    )
    return " ".join([s.text.strip() for s in segments if s.text]).strip()

async def handle_websocket(websocket):
    logging.info("Client connected")
    audio_buffer = bytearray()
    last_partial_text = ""
    is_processing = False

    try:
        async for message in websocket:
            # 1. Process binary audio stream
            if isinstance(message, bytes):
                audio_buffer.extend(message)
                
                # Check every ~1 second (32,000 bytes at 16kHz PCM16)
                if len(audio_buffer) >= 32000 and not is_processing:
                    is_processing = True
                    
                    # Convert accumulated PCM data without clearing the buffer
                    audio_snapshot = bytes(audio_buffer)
                    audio_np = np.frombuffer(audio_snapshot, dtype=np.int16).astype(np.float32) / 32768.0
                    
                    # Run off-thread
                    partial_text = await asyncio.to_thread(run_transcription, audio_np, False)
                    
                    if partial_text and partial_text != last_partial_text:
                        last_partial_text = partial_text
                        await websocket.send(json.dumps({
                            "type": "partial",
                            "text": partial_text
                        }))
                    
                    is_processing = False

            # 2. Process EOS (Stop Button)
            elif isinstance(message, str):
                if message == "EOS":
                    logging.info("EOS signal received. Running final transcription pass...")
                    
                    if len(audio_buffer) > 0:
                        audio_np = np.frombuffer(bytes(audio_buffer), dtype=np.int16).astype(np.float32) / 32768.0
                        # Disable VAD on final pass so no spoke words are stripped out
                        final_text = await asyncio.to_thread(run_transcription, audio_np, True)
                    else:
                        final_text = last_partial_text

                    await websocket.send(json.dumps({
                        "type": "final",
                        "text": final_text or last_partial_text
                    }))
                    
                    audio_buffer.clear()
                    await websocket.close()
                    break

    except websockets.exceptions.ConnectionClosed:
        logging.info("Client disconnected")
    except Exception as e:
        logging.error(f"Error handling WebSocket: {e}")

async def main():
    server = await websockets.serve(handle_websocket, "0.0.0.0", 6009)
    logging.info("Faster-Whisper ASR listening on port 6009...")
    await asyncio.get_running_loop().create_future()

if __name__ == "__main__":
    asyncio.run(main())