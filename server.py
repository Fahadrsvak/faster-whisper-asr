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

def transcribe_chunk(audio_np, is_final=False):
    """Off-thread transcription call with relaxed VAD parameters."""
    segments, _ = model.transcribe(
        audio_np,
        beam_size=1,
        language="en",
        vad_filter=not is_final, # Turn off VAD on final pass so it forces transcription
        vad_parameters=dict(threshold=0.3, min_silence_duration_ms=500)
    )
    return " ".join([segment.text.strip() for segment in segments]).strip()

async def handle_websocket(websocket):
    logging.info("Client connected")
    audio_buffer = bytearray()
    full_audio = bytearray()
    last_partial_text = ""

    try:
        async for message in websocket:
            # 1. Process binary PCM16 audio chunks
            if isinstance(message, bytes):
                audio_buffer.extend(message)
                full_audio.extend(message)
                
                # Transcribe roughly every 1.5 seconds (48,000 bytes at 16kHz PCM16)
                # Larger window helps Whisper identify spoken words over background silence
                if len(audio_buffer) >= 48000:
                    pcm_data = audio_buffer.copy()
                    audio_buffer.clear() # Clear intermediate buffer to prevent exponential CPU load
                    
                    audio_np = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
                    partial_text = await asyncio.to_thread(transcribe_chunk, audio_np, False)
                    
                    if partial_text and partial_text != last_partial_text:
                        last_partial_text = partial_text
                        await websocket.send(json.dumps({
                            "type": "partial",
                            "text": partial_text
                        }))

            # 2. Process string commands (EOS)
            elif isinstance(message, str):
                if message == "EOS":
                    logging.info("EOS signal received. Running final transcription pass...")
                    
                    if len(full_audio) > 0:
                        audio_np = np.frombuffer(full_audio, dtype=np.int16).astype(np.float32) / 32768.0
                        final_text = await asyncio.to_thread(transcribe_chunk, audio_np, True)
                    else:
                        final_text = last_partial_text

                    await websocket.send(json.dumps({
                        "type": "final",
                        "text": final_text or last_partial_text
                    }))
                    
                    audio_buffer.clear()
                    full_audio.clear()
                    await websocket.close()
                    break

    except websockets.exceptions.ConnectionClosed:
        logging.info("Client disconnected")
    except Exception as e:
        logging.error(f"Error: {e}")

async def main():
    server = await websockets.serve(handle_websocket, "0.0.0.0", 6009)
    logging.info("Faster-Whisper ASR WebSocket server listening on port 6009...")
    await asyncio.get_running_loop().create_future()

if __name__ == "__main__":
    asyncio.run(main())