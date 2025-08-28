import os
import json
import logging
import threading
import tempfile
import numpy as np
import torch
import soundfile as sf

from whisper_live.backend.base import ServeClientBase


class ServeClientParakeet(ServeClientBase):
    """Parakeet backend for WhisperLive-compatible streaming ASR"""

    # A lock to ensure thread-safe access to the shared preloaded model.
    MODEL_LOCK = threading.Lock()

    def __init__(
        self,
        websocket,
        preloaded_model,
        task="transcribe",
        device=None,
        language=None,
        client_uid=None,
        initial_prompt=None,
        vad_parameters=None,
        use_vad=True,
        send_last_n_segments=10,
        no_speech_thresh=0.45,
        clip_audio=False,
        same_output_threshold=7,
        cache_path="~/.cache/whisper-live/",
        translation_queue=None,
    ):
        """
        Initialize Parakeet client for streaming ASR.

        Args:
            websocket: WebSocket connection.
            preloaded_model: A pre-loaded Parakeet ASR model instance. This is required.
            task: Task type (transcribe).
            device: Device to use (cuda/cpu).
            language: Language code.
            client_uid: Unique client identifier.
            initial_prompt: Not used for Parakeet.
            vad_parameters: VAD parameters dictionary.
            use_vad: Whether to use VAD.
            send_last_n_segments: Number of segments to send.
            no_speech_thresh: No speech threshold.
            clip_audio: Whether to clip audio.
            same_output_threshold: Threshold for same output.
            cache_path: Cache directory path.
            translation_queue: Queue for translation.
        """
        super().__init__(
            client_uid,
            websocket,
            send_last_n_segments,
            no_speech_thresh,
            clip_audio,
            same_output_threshold,
            translation_queue
        )

        self.cache_path = cache_path
        self.model_name = "nvidia/parakeet-tdt-0.6b-v3"  # Model is fixed to Parakeet v3
        self.language = "de" if language is None else language
        self.task = task
        self.initial_prompt = initial_prompt
        self.vad_parameters = vad_parameters or {"onset": 0.5}
        self.use_vad = use_vad

        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logging.info(f"[Parakeet] Using device: {self.device}")

        # Parakeet specific parameters
        self.sample_rate = 16000
        self.min_chunk_duration = 1.0  # Minimum chunk size in seconds
        self.min_chunk_samples = int(
            self.sample_rate * self.min_chunk_duration)
        self.temp_dir = tempfile.mkdtemp()

        # Use the preloaded model provided by the server
        if preloaded_model is None:
            raise ValueError("A preloaded Parakeet model must be provided.")
        self.transcriber = preloaded_model
        logging.info("[Parakeet] Using pre-loaded model from server.")

        # Start transcription thread
        try:
            self.trans_thread = threading.Thread(target=self.speech_to_text)
            self.trans_thread.start()
        except Exception as e:
            logging.error(
                f"[Parakeet] Failed to start transcription thread: {e}")

        # Send ready message - this is crucial to prevent client from hanging
        try:
            self.websocket.send(json.dumps({
                "uid": self.client_uid,
                "message": self.SERVER_READY,
                "backend": "parakeet"
            }))
            logging.info(f"[Parakeet] Client {self.client_uid} ready")
        except Exception as e:
            logging.error(
                f"[Parakeet] Failed to send SERVER_READY message: {e}")

    def add_frames(self, frame_bytes):
        """
        Handles incoming audio frames from the client.
        """
        self.frames += frame_bytes
        self.frames_np = np.frombuffer(self.frames, dtype=np.float32)

    def transcribe_audio_batch(self, input_samples):
        """
        Transcribe a batch of audio clips using the shared Parakeet model.
        """
        with ServeClientParakeet.MODEL_LOCK:
            batch_results = []
            temp_paths = []
            try:
                for input_sample in input_samples:
                    temp_fd, temp_path = tempfile.mkstemp(
                        suffix='.wav', dir=self.temp_dir)
                    os.close(temp_fd)
                    temp_paths.append(temp_path)

                    if input_sample.dtype != np.float32:
                        input_sample = input_sample.astype(np.float32)

                    max_val = np.max(np.abs(input_sample))
                    if max_val > 1.0:
                        input_sample = input_sample / max_val

                    sf.write(temp_path, input_sample, self.sample_rate)

                if not temp_paths:
                    return []

                outputs = self.transcriber.transcribe(temp_paths)

                if outputs and len(outputs) == len(input_samples):
                    for i, output in enumerate(outputs):
                        transcript_text = (output.text if hasattr(output, 'text') else str(output)).strip()
                        if transcript_text:
                            segment = type('Segment', (), {
                                'text': transcript_text,
                                'start': 0.0,
                                'end': len(input_samples[i]) / self.sample_rate,
                                'no_speech_prob': 0.0, 'tokens': [], 'temperature': 0.0,
                                'avg_logprob': 0.0, 'compression_ratio': 0.0, 'words': None
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

    def transcribe_audio(self, input_sample):
        """
        Transcribe a single audio clip using the shared Parakeet model.
        """
        with ServeClientParakeet.MODEL_LOCK:
            result = []
            temp_fd, temp_path = tempfile.mkstemp(suffix='.wav', dir=self.temp_dir)
            os.close(temp_fd)

            try:
                if input_sample.dtype != np.float32:
                    input_sample = input_sample.astype(np.float32)

                max_val = np.max(np.abs(input_sample))
                if max_val > 1.0:
                    input_sample = input_sample / max_val
                elif max_val < 0.01:
                    return []

                sf.write(temp_path, input_sample, self.sample_rate)
                outputs = self.transcriber.transcribe([temp_path])

                if outputs and len(outputs) > 0:
                    transcript_text = (outputs[0].text if hasattr(outputs[0], 'text') else str(outputs[0])).strip()
                    if transcript_text:
                        segment = type('Segment', (), {
                            'text': transcript_text,
                            'start': 0.0,
                            'end': len(input_sample) / self.sample_rate,
                            'no_speech_prob': 0.0, 'tokens': [], 'temperature': 0.0,
                            'avg_logprob': 0.0, 'compression_ratio': 0.0, 'words': None
                        })()
                        result = [segment]
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            return result

    def handle_transcription_output(self, result, duration):
        """
        Handle the transcription output.
        """
        segments = []
        if len(result):
            self.t_start = None
            last_segment = self.update_segments(result, duration)
            segments = self.prepare_segments(last_segment)

        if len(segments):
            self.send_transcription_to_client(segments)

    def cleanup(self):
        """Clean up resources."""
        super().cleanup()
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            import shutil
            try:
                shutil.rmtree(self.temp_dir)
            except OSError as e:
                logging.error(f"[Parakeet] Error cleaning up temp directory {self.temp_dir}: {e}")

