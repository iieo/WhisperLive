import os
import time
import threading
import queue
import json
import logging
import tempfile
import shutil
from typing import List, Optional

import numpy as np
import soundfile as sf
from websockets.sync.server import serve
from websockets.exceptions import ConnectionClosed

from whisper_live.backend.base import ServeClientBase
from whisper_live.backend.parakeet_backend import ServeClientParakeet
from whisper_live.backend.batch_manager import BatchManager

logging.basicConfig(level=logging.INFO)


class ClientManager:
    def __init__(self, max_clients=100, max_connection_time=600):
        self.clients = {}
        self.start_times = {}
        self.max_clients = max_clients
        self.max_connection_time = max_connection_time

    def add_client(self, websocket, client):
        self.clients[websocket] = client
        self.start_times[websocket] = time.time()

    def get_client(self, websocket):
        return self.clients.get(websocket, False)

    def remove_client(self, websocket):
        client = self.clients.pop(websocket, None)
        if client:
            client.cleanup()
        self.start_times.pop(websocket, None)

    def get_wait_time(self):
        if not self.start_times:
            return 0

        wait_time = min(
            self.max_connection_time - (time.time() - start_time)
            for start_time in self.start_times.values()
        )
        return max(0, wait_time) / 60

    def is_server_full(self, websocket, options):
        if len(self.clients) >= self.max_clients:
            wait_time = self.get_wait_time()
            response = {"uid": options["uid"],
                        "status": "WAIT", "message": wait_time}
            websocket.send(json.dumps(response))
            return True
        return False

    def is_client_timeout(self, websocket):
        elapsed_time = time.time() - self.start_times[websocket]
        if elapsed_time >= self.max_connection_time:
            self.clients[websocket].disconnect()
            logging.warning(
                f"Client with uid '{self.clients[websocket].client_uid}' disconnected due to overtime.")
            return True
        return False


class TranscriptionServer:
    RATE = 16000
    MODEL_LOCK = threading.Lock()

    def __init__(self):
        self.client_manager = None
        self.use_vad = True
        self.single_model = False
        self.preloaded_parakeet_model = None
        self.batch_manager = None
        self.temp_dir = tempfile.mkdtemp()

    def _preload_parakeet_model(self):
        try:
            import torch
            import nemo.collections.asr as nemo_asr
            logging.info("[Server] Pre-loading Parakeet model...")
            model = nemo_asr.models.ASRModel.from_pretrained(
                model_name="nvidia/parakeet-tdt-0.6b-v3")
            if torch.cuda.is_available():
                logging.info("[Server] Moving Parakeet model to CUDA...")
                model = model.cuda()
            model.eval()
            self.preloaded_parakeet_model = model
            logging.info("[Server] Parakeet model pre-loaded successfully")
        except Exception as e:
            logging.error(f"[Server] Failed to pre-load Parakeet model: {e}")
            self.preloaded_parakeet_model = None

    def initialize_client(self, websocket, options):
        client: Optional[ServeClientBase] = None
        translation_queue = None
        translation_client = None
        translation_thread = None

        if options.get("enable_translation", False):
            target_language = options.get("target_language", "de")
            translation_queue = queue.Queue()
            from whisper_live.backend.translation_backend import ServeClientTranslation
            translation_client = ServeClientTranslation(
                client_uid=options["uid"],
                websocket=websocket,
                translation_queue=translation_queue,
                target_language=target_language,
                send_last_n_segments=options.get("send_last_n_segments", 10)
            )
            translation_thread = threading.Thread(
                target=translation_client.speech_to_text, daemon=True)
            translation_thread.start()
            logging.info(
                f"Translation enabled for client {options['uid']} with target language: {target_language}")

        try:
            if self.single_model and not self.batch_manager:
                raise RuntimeError(
                    "Server is in single_model mode but BatchManager is not initialized.")

            client = ServeClientParakeet(
                websocket,
                language=options.get("language"),
                task=options.get("task", "transcribe"),
                client_uid=options["uid"],
                vad_parameters=options.get("vad_parameters"),
                use_vad=self.use_vad,
                send_last_n_segments=options.get("send_last_n_segments", 10),
                no_speech_thresh=options.get("no_speech_thresh", 0.45),
                clip_audio=options.get("clip_audio", False),
                same_output_threshold=options.get("same_output_threshold", 7),
                translation_queue=translation_queue,
                batch_manager=self.batch_manager
            )
            logging.info("Running Parakeet backend.")
        except Exception as e:
            logging.error(f"Error initializing Parakeet backend: {e}")
            try:
                websocket.send(json.dumps(
                    {"uid": options["uid"], "status": "ERROR", "message": str(e)}))
            except ConnectionClosed:
                pass
            return

        if translation_client:
            client.translation_client = translation_client
            client.translation_thread = translation_thread

        self.client_manager.add_client(websocket, client)

    def get_audio_from_websocket(self, websocket):
        frame_data = websocket.recv()
        if frame_data == b"END_OF_AUDIO":
            return False
        return np.frombuffer(frame_data, dtype=np.float32)

    def handle_new_connection(self, websocket):
        try:
            logging.info("New client connected")
            options = json.loads(websocket.recv())
            self.use_vad = options.get('use_vad', True)
            if self.client_manager.is_server_full(websocket, options):
                websocket.close()
                return False
            self.initialize_client(websocket, options)
            return True
        except (json.JSONDecodeError, ConnectionClosed) as e:
            logging.error(f"Failed to initialize connection: {e}")
            return False
        except Exception as e:
            logging.error(f"Error during new connection initialization: {e}")
            return False

    def process_audio_frames(self, websocket):
        client = self.client_manager.get_client(websocket)
        if not client:
            return False

        frame_np = self.get_audio_from_websocket(websocket)
        if frame_np is False:
            return False

        client.add_frames(frame_np)
        return True

    def recv_audio(self, websocket):
        if not self.handle_new_connection(websocket):
            return

        try:
            while not self.client_manager.is_client_timeout(websocket):
                if not self.process_audio_frames(websocket):
                    break
        except ConnectionClosed:
            logging.info("Connection closed by client")
        except Exception as e:
            logging.error(f"Unexpected error for client: {e}")
        finally:
            self.cleanup(websocket)

    def transcribe_audio_batch(self, input_samples: List[np.ndarray]) -> List[List]:
        with self.MODEL_LOCK:
            batch_results = []
            temp_paths = []
            try:
                for input_sample in input_samples:
                    temp_fd, temp_path = tempfile.mkstemp(
                        suffix='.wav', dir=self.temp_dir)
                    os.close(temp_fd)
                    temp_paths.append(temp_path)

                    input_sample = input_sample.astype(np.float32)
                    max_val = np.max(np.abs(input_sample))
                    if max_val > 1.0:
                        input_sample = input_sample / max_val
                    sf.write(temp_path, input_sample, self.RATE)

                if not temp_paths:
                    return []

                outputs = self.preloaded_parakeet_model.transcribe(temp_paths)

                if outputs and len(outputs) == len(input_samples):
                    for i, output in enumerate(outputs):
                        text = getattr(output, 'text', str(output)).strip()
                        if text:
                            segment = type('Segment', (), {
                                'text': text, 'start': 0.0, 'end': len(input_samples[i]) / self.RATE,
                                'no_speech_prob': 0.0
                            })()
                            batch_results.append([segment])
                        else:
                            batch_results.append([])
                else:
                    batch_results = [[] for _ in input_samples]
            finally:
                for path in temp_paths:
                    if os.path.exists(path):
                        os.unlink(path)
            return batch_results

    def _cleanup_server_resources(self):
        logging.info("Cleaning up server resources...")
        if self.batch_manager:
            self.batch_manager.stop()
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                logging.info(
                    f"Cleaned up server temp directory {self.temp_dir}")
            except OSError as e:
                logging.error(
                    f"Error cleaning up server temp directory {self.temp_dir}: {e}")

    def run(self, host, port=8005, single_model=False, max_clients=4, max_connection_time=600, cache_path="~/.cache/whisper-live/"):
        self.cache_path = os.path.expanduser(cache_path)
        self.client_manager = ClientManager(max_clients, max_connection_time)

        if single_model:
            self.single_model = True
            self._preload_parakeet_model()
            if self.preloaded_parakeet_model:
                self.batch_manager = BatchManager(
                    batch_processor=self.transcribe_audio_batch,
                    batch_size=max_clients,
                    batch_timeout=0.1
                )

        try:
            with serve(self.recv_audio, host, port) as server:
                logging.info(
                    f"Parakeet Transcription Server is running on {host}:{port}")
                server.serve_forever()
        finally:
            self._cleanup_server_resources()

    def cleanup(self, websocket):
        client = self.client_manager.get_client(websocket)
        if client:
            logging.info(f"Cleaning up client {client.client_uid}")
            if hasattr(client, 'translation_thread') and client.translation_thread.is_alive():
                client.translation_thread.join(timeout=2.0)
            self.client_manager.remove_client(websocket)
        try:
            websocket.close()
        except ConnectionClosed:
            pass
