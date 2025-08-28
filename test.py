import asyncio
from uuid import uuid4
import websockets
import json
import soundfile as sf
import numpy as np

URL = "wss://stt.iieo.de"  # or "wss://stt.iieo.de"
AUDIO_FILE = "test.wav"  # 16kHz, mono, PCM16

CHUNK_SIZE = 1024  # samples per chunk

# Convert PCM16 to Float32 bytes


def pcm16_to_float32_bytes(data):
    float32 = (data / 32768.0).astype(np.float32)
    return float32.tobytes()


async def stream_audio(client_id):
    uuid = f"python_client_{client_id}_{uuid4()}"
    try:
        async with websockets.connect(URL) as ws:
            print(f"Client {client_id} connected")

            # Send initial handshake
            init_msg = json.dumps({
                "uid": uuid,
                "enable_translation": "de",
                "task": "transcribe",
                "translate": False,
                "model": "small",
                "use_vad": False
            })
            await ws.send(init_msg)

            # Load WAV file
            data, samplerate = sf.read(AUDIO_FILE, dtype='int16')
            if len(data.shape) > 1:
                data = data[:, 0]  # take first channel if stereo

            # Stream audio in chunks
            for start in range(0, len(data), CHUNK_SIZE):
                chunk = data[start:start + CHUNK_SIZE]
                await ws.send(pcm16_to_float32_bytes(chunk))
                # await asyncio.sleep(CHUNK_SIZE / samplerate)  # simulate real-time

            # Optional: send a final empty chunk or "end" message if server requires
            # await ws.send(b'')

            # Listen for transcriptions
            async for message in ws:
                print(f"Client {client_id} received: {message}")

    except Exception as e:
        print(f"Client {client_id} error: {e}")


async def main():
    tasks = [stream_audio(i) for i in range(1, 4)]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
