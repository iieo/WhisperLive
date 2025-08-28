import os
import time
import threading
import queue
import json
import functools
import logging
from typing import List, Optional

import numpy as np
from websockets.sync.server import serve
from websockets.exceptions import ConnectionClosed
from whisper_live.backend.base import ServeClientBase
from whisper_live.backend.parakeet_backend import ServeClientParakeet

logging.basicConfig(level=logging.INFO)


class ClientManager:
    def __init__(self, max_clients=100, max_connection_time=600):
        """
        Initializes the ClientManager with specified limits on client connections and connection durations.

        Args:
            max_clients (int, optional): The maximum number of simultaneous client connections allowed. Defaults to 4.
            max_connection_time (int, optional): The maximum duration (in seconds) a client can stay connected. Defaults
                                                 to 600 seconds (10 minutes).
        """
        self.clients = {}
        self.start_times = {}
        self.max_clients = max_clients
        self.max_connection_time = max_connection_time

    def add_client(self, websocket, client):
        """
        Adds a client and their connection start time to the tracking dictionaries.

        Args:
            websocket: The websocket associated with the client to add.
            client: The client object to be added and tracked.
        """
        self.clients[websocket] = client
        self.start_times[websocket] = time.time()

    def get_client(self, websocket):
        """
        Retrieves a client associated with the given websocket.

        Args:
            websocket: The websocket associated with the client to retrieve.

        Returns:
            The client object if found, False otherwise.
        """
        if websocket in self.clients:
            return self.clients[websocket]
        return False

    def remove_client(self, websocket):
        """
        Removes a client and their connection start time from the tracking dictionaries. Performs cleanup on the
        client if necessary.

        Args:
            websocket: The websocket associated with the client to be removed.
        """
        client = self.clients.pop(websocket, None)
        if client:
            client.cleanup()
        self.start_times.pop(websocket, None)

    def get_wait_time(self):
        """
        Calculates the estimated wait time for new clients based on the remaining connection times of current clients.

        Returns:
            The estimated wait time in minutes for new clients to connect. Returns 0 if there are available slots.
        """
        wait_time = None
        for start_time in self.start_times.values():
            current_client_time_remaining = self.max_connection_time - \
                (time.time() - start_time)
            if wait_time is None or current_client_time_remaining < wait_time:
                wait_time = current_client_time_remaining
        return wait_time / 60 if wait_time is not None else 0

    def is_server_full(self, websocket, options):
        """
        Checks if the server is at its maximum client capacity and sends a wait message to the client if necessary.

        Args:
            websocket: The websocket of the client attempting to connect.
            options: A dictionary of options that may include the client's unique identifier.

        Returns:
            True if the server is full, False otherwise.
        """
        if len(self.clients) >= self.max_clients:
            wait_time = self.get_wait_time()
            response = {"uid": options["uid"],
                        "status": "WAIT", "message": wait_time}
            websocket.send(json.dumps(response))
            return True
        return False

    def is_client_timeout(self, websocket):
        """
        Checks if a client has exceeded the maximum allowed connection time and disconnects them if so, issuing a warning.

        Args:
            websocket: The websocket associated with the client to check.

        Returns:
            True if the client's connection time has exceeded the maximum limit, False otherwise.
        """
        elapsed_time = time.time() - self.start_times[websocket]
        if elapsed_time >= self.max_connection_time:
            self.clients[websocket].disconnect()
            logging.warning(
                f"Client with uid '{self.clients[websocket].client_uid}' disconnected due to overtime.")
            return True
        return False


class TranscriptionServer:
    RATE = 16000

    def __init__(self):
        self.client_manager = None
        self.use_vad = True
        self.single_model = False
        self.preloaded_parakeet_model = None

    def _preload_parakeet_model(self):
        """Pre-load the parakeet model at server startup"""
        try:
            import torch
            import nemo.collections.asr as nemo_asr

            logging.info("[Server] Pre-loading Parakeet model...")

            # Load Parakeet model from NVIDIA NGC
            model = nemo_asr.models.ASRModel.from_pretrained(
                model_name="nvidia/parakeet-tdt-0.6b-v3"
            )

            # Move to GPU if available
            if torch.cuda.is_available():
                logging.info("[Server] Moving Parakeet model to CUDA...")
                model = model.cuda()

            # Set to evaluation mode
            model.eval()

            self.preloaded_parakeet_model = model
            logging.info("[Server] Parakeet model pre-loaded successfully")

        except Exception as e:
            logging.error(f"[Server] Failed to pre-load Parakeet model: {e}")
            self.preloaded_parakeet_model = None

    def initialize_client(self, websocket, options):
        client: Optional[ServeClientBase] = None

        # Check if client wants translation
        enable_translation = options.get("enable_translation", False)

        # Create translation queue if translation is enabled
        translation_queue = None
        translation_client = None
        translation_thread = None

        if enable_translation:
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

            # Start translation thread
            translation_thread = threading.Thread(
                target=translation_client.speech_to_text,
                daemon=True
            )
            translation_thread.start()

            logging.info(
                f"Translation enabled for client {options['uid']} with target language: {target_language}")

        # Handle Parakeet backend
        try:
            model_name = "nvidia/parakeet-tdt-0.6b-v3"
            client = ServeClientParakeet(
                websocket,
                language=options.get("language"),
                task=options.get("task", "transcribe"),
                client_uid=options["uid"],
                model=model_name,
                initial_prompt=options.get("initial_prompt"),
                vad_parameters=options.get("vad_parameters"),
                use_vad=self.use_vad,
                single_model=self.single_model,
                send_last_n_segments=options.get(
                    "send_last_n_segments", 10),
                no_speech_thresh=options.get("no_speech_thresh", 0.45),
                clip_audio=options.get("clip_audio", False),
                same_output_threshold=options.get(
                    "same_output_threshold", 10),
                cache_path=self.cache_path,
                translation_queue=translation_queue,
                preloaded_model=self.preloaded_parakeet_model
            )
            logging.info("Running Parakeet backend.")
        except Exception as e:
            logging.error(f"Error initializing Parakeet backend: {e}")
            try:
                websocket.send(json.dumps({
                    "uid": options["uid"],
                    "status": "ERROR",
                    "message": f"Parakeet backend failed to initialize: {str(e)}"
                }))
            except ConnectionClosed:
                pass  # Client may have already disconnected
            return

        if translation_client:
            client.translation_client = translation_client
            client.translation_thread = translation_thread

        self.client_manager.add_client(websocket, client)

    def get_audio_from_websocket(self, websocket):
        """
        Receives audio buffer from websocket and creates a numpy array out of it.

        Args:
            websocket: The websocket to receive audio from.

        Returns:
            A numpy array containing the audio or False if the connection is closing.
        """
        frame_data = websocket.recv()
        if frame_data == b"END_OF_AUDIO":
            return False
        return np.frombuffer(frame_data, dtype=np.float32)

    def handle_new_connection(self, websocket):
        try:
            logging.info("New client connected")
            options = websocket.recv()
            options = json.loads(options)

            self.use_vad = options.get('use_vad', True)
            if self.client_manager.is_server_full(websocket, options):
                websocket.close()
                return False  # Indicates that the connection should not continue

            self.initialize_client(websocket, options)
            return True
        except json.JSONDecodeError:
            logging.error("Failed to decode JSON from client")
            return False
        except ConnectionClosed:
            logging.info("Connection closed by client during initialization")
            return False
        except Exception as e:
            logging.error(
                f"Error during new connection initialization: {str(e)}")
            return False

    def process_audio_frames(self, websocket):
        frame_np = self.get_audio_from_websocket(websocket)
        client = self.client_manager.get_client(websocket)
        if frame_np is False:
            return False

        client.add_frames(frame_np)
        return True

    def recv_audio(self, websocket):
        """
        Receive audio chunks from a client in an infinite loop.

        Continuously receives audio frames from a connected client
        over a WebSocket connection. It adds the audio frames to the client's
        audio data for ASR.
        If the maximum number of clients is reached, the method sends a
        "WAIT" status to the client, indicating that they should wait
        until a slot is available.
        If a client's connection exceeds the maximum allowed time, it will
        be disconnected, and the client's resources will be cleaned up.

        Args:
            websocket (WebSocket): The WebSocket connection for the client.

        Raises:
            Exception: If there is an error during the audio frame processing.
        """
        if not self.handle_new_connection(websocket):
            return

        try:
            while not self.client_manager.is_client_timeout(websocket):
                if not self.process_audio_frames(websocket):
                    break
        except ConnectionClosed:
            logging.info("Connection closed by client")
        except Exception as e:
            logging.error(f"Unexpected error: {str(e)}")
        finally:
            if self.client_manager.get_client(websocket):
                self.cleanup(websocket)
                websocket.close()
            del websocket

    def run(self,
            host,
            port=8005,
            single_model=False,
            max_clients=4,
            max_connection_time=600,
            cache_path="~/.cache/whisper-live/"):
        """
        Run the transcription server.

        Args:
            host (str): The host address to bind the server.
            port (int): The port number to bind the server.
            single_model (bool): If True, pre-loads the Parakeet model on startup.
            max_clients (int): The maximum number of simultaneous client connections allowed.
            max_connection_time (int): The maximum duration (in seconds) a client can stay connected.
            cache_path (str): Path to cache directory.
        """
        self.cache_path = cache_path
        self.client_manager = ClientManager(max_clients, max_connection_time)

        # Pre-load parakeet model if single_model mode is enabled
        if single_model:
            self.single_model = True
            self._preload_parakeet_model()

        with serve(
            self.recv_audio,
            host,
            port
        ) as server:
            logging.info(
                f"Parakeet Transcription Server is running on {host}:{port}")
            server.serve_forever()

    def cleanup(self, websocket):
        """
        Cleans up resources associated with a given client's websocket.

        Args:
            websocket: The websocket associated with the client to be cleaned up.
        """
        client = self.client_manager.get_client(websocket)
        if client:
            if hasattr(client, 'translation_client') and client.translation_client:
                client.translation_client.cleanup()

            # Wait for translation thread to finish
            if hasattr(client, 'translation_thread') and client.translation_thread:
                client.translation_thread.join(timeout=2.0)
            self.client_manager.remove_client(websocket)
