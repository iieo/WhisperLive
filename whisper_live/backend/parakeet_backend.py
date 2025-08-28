import json
import logging
import threading
import numpy as np
import torch

from whisper_live.backend.base import ServeClientBase


class ServeClientParakeet(ServeClientBase):
    """
    Parakeet backend for WhisperLive-compatible streaming ASR.
    This client relies on a BatchManager for transcription when in single_model mode.
    """

    def __init__(
        self,
        websocket,
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
        batch_manager=None,
    ):
        """
        Initialize Parakeet client for streaming ASR.
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

        self.language = "de" if language is None else language
        self.task = task
        self.vad_parameters = vad_parameters or {"onset": 0.5}
        self.use_vad = use_vad

        # Check for BatchManager
        self.batch_manager = batch_manager
        if self.batch_manager is None:
            raise ValueError(
                "ServeClientParakeet requires a BatchManager instance.")
        logging.info("[Parakeet] Using BatchManager for transcription.")

        # Start transcription thread
        try:
            self.trans_thread = threading.Thread(target=self.speech_to_text)
            self.trans_thread.start()
        except Exception as e:
            logging.error(
                f"[Parakeet] Failed to start transcription thread: {e}")

        # Send ready message
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

    def transcribe_audio(self, input_sample):
        """
        Submits an audio sample to the batch manager for transcription.
        """
        # A heuristic to avoid sending empty or near-empty audio to the model
        if np.max(np.abs(input_sample)) < 0.01:
            return []
        return self.batch_manager.submit_job(input_sample)

    def handle_transcription_output(self, result, duration):
        """
        Handle the transcription output from the BatchManager.
        """
        segments = []
        if result and len(result):
            self.t_start = None
            last_segment = self.update_segments(result, duration)
            segments = self.prepare_segments(last_segment)

        if len(segments):
            self.send_transcription_to_client(segments)
