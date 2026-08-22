import asyncio
import json
import logging
import time
import numpy as np
import websockets
from faster_whisper import WhisperModel

logging.basicConfig(level=logging.INFO)

# Load model with int8 quantization for optimal CPU performance
MODEL_SIZE = "base.en"
logging.info(f"Loading Faster-Whisper model ({MODEL_SIZE})...")
model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8", cpu_threads=4)
logging.info("Faster-Whisper initialized successfully.")

def transcribe_audio(audio_np):
    """Off-thread transcription call with VAD completely disabled."""
    start_time = time.perf_counter()
    
    # vad_filter=False prevents Silero VAD from stripping audio
    segments, info = model.transcribe(
        audio_np,
        beam_size=1,            # Greedy decoding for max CPU speed
        language="en",
        vad_filter=False        # DISABLED: Processes raw audio directly
    )
    
    text = " ".join([segment.text.strip() for segment in segments if segment.text]).strip()
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    
    return text, elapsed_ms, info.duration

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
                    
                    audio_snapshot = bytes(audio_buffer)
                    audio_np = np.frombuffer(audio_snapshot, dtype=np.int16).astype(np.float32) / 32768.0
                    
                    # Run transcription pass off-thread
                    partial_text, inference_ms, audio_dur = await asyncio.to_thread(transcribe_audio, audio_np)
                    
                    logging.info(
                        f"[PARTIAL] Audio: {audio_dur:.2f}s | "
                        f"Inference: {inference_ms:.1f}ms | "
                        f"Text: '{partial_text}'"
                    )
                    
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
                    logging.info("EOS signal received. Executing final transcription pass...")
                    
                    if len(audio_buffer) > 0:
                        audio_np = np.frombuffer(bytes(audio_buffer), dtype=np.int16).astype(np.float32) / 32768.0
                        final_text, inference_ms, audio_dur = await asyncio.to_thread(transcribe_audio, audio_np)
                        
                        logging.info(
                            f"[FINAL] Audio: {audio_dur:.2f}s | "
                            f"Inference: {inference_ms:.1f}ms | "
                            f"Text: '{final_text}'"
                        )
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
    logging.info("Faster-Whisper ASR WebSocket server listening on port 6009...")
    await asyncio.get_running_loop().create_future()

if __name__ == "__main__":
    asyncio.run(main())