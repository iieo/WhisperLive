import os
import json
import logging
import threading
import time
import traceback
import tempfile
import numpy as np
import torch
import soundfile as sf
import nemo.collections.asr as nemo_asr

from whisper_live.backend.base import ServeClientBase


class ServeClientParakeet(ServeClientBase):
    """Parakeet backend for WhisperLive-compatible streaming ASR"""

    SINGLE_MODEL = None
    SINGLE_MODEL_LOCK = threading.Lock()

    def __init__(
        self,
        websocket,
        task="transcribe",
        device=None,
        language=None,
        client_uid=None,
        model="nvidia/parakeet-tdt-0.6b-v3",
        initial_prompt=None,
        vad_parameters=None,
        use_vad=True,
        single_model=False,
        send_last_n_segments=10,
        no_speech_thresh=0.45,
        clip_audio=False,
        same_output_threshold=7,
        cache_path="~/.cache/whisper-live/",
        translation_queue=None,
        preloaded_model=None,
    ):
        """
        Initialize Parakeet client for streaming ASR.

        Args:
            websocket: WebSocket connection
            task: Task type (transcribe)
            device: Device to use (cuda/cpu)
            language: Language code
            client_uid: Unique client identifier
            model: Parakeet model name/path
            initial_prompt: Not used for Parakeet
            vad_parameters: VAD parameters dictionary
            use_vad: Whether to use VAD
            single_model: Whether to share model across clients
            send_last_n_segments: Number of segments to send
            no_speech_thresh: No speech threshold
            clip_audio: Whether to clip audio
            same_output_threshold: Threshold for same output
            cache_path: Cache directory path
            translation_queue: Queue for translation
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
        self.model_name = model
        self.language = "de" if language is None else language
        self.task = task
        self.initial_prompt = initial_prompt
        self.vad_parameters = vad_parameters or {"onset": 0.5}
        self.use_vad = use_vad
        self.single_model = single_model

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

        # Load model
        model_loaded = False
        try:
            if preloaded_model is not None:
                # Use pre-loaded model from server
                self.transcriber = preloaded_model
                logging.info("[Parakeet] Using pre-loaded model from server")
                print("[Parakeet] Using pre-loaded model from server")
            elif single_model:
                with ServeClientParakeet.SINGLE_MODEL_LOCK:
                    if ServeClientParakeet.SINGLE_MODEL is None:
                        self.create_model(self.device)
                        ServeClientParakeet.SINGLE_MODEL = self.transcriber
                    else:
                        self.transcriber = ServeClientParakeet.SINGLE_MODEL
            else:
                self.create_model(self.device)
            model_loaded = True
        except Exception as e:
            logging.error(f"[Parakeet] Failed to load model: {e}")
            logging.error(f"[Parakeet] Traceback: {traceback.format_exc()}")
            # Send error message but continue to send SERVER_READY to prevent client from hanging
            try:
                self.websocket.send(json.dumps({
                    "uid": self.client_uid,
                    "status": "ERROR",
                    "message": f"Failed to load Parakeet model: {str(e)}"
                }))
            except:
                pass
            # Don't return here - we still want to send SERVER_READY to prevent client hang

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

        This method overrides the base class's `add_frames` to correctly
        interpret the incoming audio chunks as raw float32 PCM data. The base
        class assumes 16-bit integer PCM, which would lead to misinterpretation
        of float32 data.

        Args:
            frame_bytes: A byte string containing the raw audio data.
        """
        # The base class accumulates bytes in `self.frames`. We do the same
        # to ensure compatibility with buffer clipping logic in the base class.
        self.frames += frame_bytes

        # Interpret the accumulated bytes as a numpy array of float32 values.
        # This assumes the client is sending 32-bit float PCM.
        self.frames_np = np.frombuffer(self.frames, dtype=np.float32)

    def create_model(self, device):
        """Load Parakeet model"""
        logging.info(f"[Parakeet] Loading model: {self.model_name}")
        print(f"[Parakeet] Loading model: {self.model_name}")

        try:
            # Load Parakeet model from NVIDIA NGC
            logging.info("[Parakeet] Calling from_pretrained...")
            print("[Parakeet] Calling from_pretrained...")
            self.transcriber = nemo_asr.models.ASRModel.from_pretrained(
                model_name=self.model_name
            )
            logging.info("[Parakeet] Model loaded from_pretrained completed")
            print("[Parakeet] Model loaded from_pretrained completed")

            # Move to device
            if device == "cuda":
                logging.info("[Parakeet] Moving model to CUDA...")
                print("[Parakeet] Moving model to CUDA...")
                self.transcriber = self.transcriber.cuda()

            # Set to evaluation mode
            logging.info("[Parakeet] Setting model to evaluation mode...")
            print("[Parakeet] Setting model to evaluation mode...")
            self.transcriber.eval()

            logging.info("[Parakeet] Model loaded successfully")
            print("[Parakeet] Model loaded successfully")

        except Exception as e:
            logging.error(f"[Parakeet] Failed to load model: {e}")
            print(f"[Parakeet] Failed to load model: {e}")
            raise

    def transcribe_audio_batch(self, input_samples):
        """
        Transcribe a batch of audio clips using Parakeet.

        Args:
            input_samples: A list of audio numpy arrays.

        Returns:
            A list of lists of segments, where each inner list corresponds
            to the transcription result for an audio clip in the input batch.
        """
        if ServeClientParakeet.SINGLE_MODEL and self.single_model:
            ServeClientParakeet.SINGLE_MODEL_LOCK.acquire()

        batch_results = []
        temp_paths = []
        try:
            # Create temporary WAV files for each sample in the batch
            for input_sample in input_samples:
                temp_fd, temp_path = tempfile.mkstemp(
                    suffix='.wav', dir=self.temp_dir)
                os.close(temp_fd)
                temp_paths.append(temp_path)

                # Ensure audio is float32
                if input_sample.dtype != np.float32:
                    input_sample = input_sample.astype(np.float32)

                # Normalize to [-1, 1]
                max_val = np.max(np.abs(input_sample))
                if max_val > 1.0:
                    input_sample = input_sample / max_val

                # Write WAV file, handle very quiet audio
                if np.max(np.abs(input_sample)) < 0.01:
                    sf.write(temp_path, np.zeros_like(
                        input_sample), self.sample_rate)
                else:
                    sf.write(temp_path, input_sample, self.sample_rate)

            # Transcribe the batch of files if any paths were created
            if temp_paths:
                outputs = self.transcriber.transcribe(temp_paths)

                if outputs and len(outputs) == len(input_samples):
                    # Process each result in the batch
                    for i, output in enumerate(outputs):
                        transcript_text = ""
                        input_sample = input_samples[i]

                        # Extract text from output
                        if hasattr(output, 'text'):
                            transcript_text = output.text.strip()
                        elif isinstance(output, str):
                            transcript_text = output.strip()
                        else:
                            transcript_text = str(output).strip()

                        if transcript_text:
                            # Create segment in Whisper-compatible format
                            segment = type('Segment', (), {
                                'text': transcript_text,
                                'start': 0.0,
                                'end': len(input_sample) / self.sample_rate,
                                'no_speech_prob': 0.0,
                                'tokens': [],
                                'temperature': 0.0,
                                'avg_logprob': 0.0,
                                'compression_ratio': 0.0,
                                'words': None
                            })()
                            batch_results.append([segment])
                        else:
                            # Append an empty list if no transcription
                            batch_results.append([])
                else:
                    # If transcription fails or returns unexpected results,
                    # create empty results for the whole batch.
                    batch_results = [[] for _ in input_samples]

        finally:
            # Clean up all temp files
            for path in temp_paths:
                if os.path.exists(path):
                    os.unlink(path)

            if ServeClientParakeet.SINGLE_MODEL and self.single_model:
                ServeClientParakeet.SINGLE_MODEL_LOCK.release()

        return batch_results

    def transcribe_audio(self, input_sample):
        """
        Transcribe audio using Parakeet

        Args:
            input_sample: Audio numpy array

        Returns:
            List of segments with transcription results
        """
        if ServeClientParakeet.SINGLE_MODEL and self.single_model:
            ServeClientParakeet.SINGLE_MODEL_LOCK.acquire()

        result = []
        try:
            # Create temporary WAV file
            temp_fd, temp_path = tempfile.mkstemp(
                suffix='.wav', dir=self.temp_dir)
            os.close(temp_fd)

            # Ensure audio is float32 and normalized
            if input_sample.dtype != np.float32:
                input_sample = input_sample.astype(np.float32)

            # Normalize to [-1, 1]
            max_val = np.max(np.abs(input_sample))
            if max_val > 1.0:
                input_sample = input_sample / max_val
            elif max_val < 0.01:  # Very quiet audio
                return []

            # Write WAV file
            sf.write(temp_path, input_sample, self.sample_rate)

            # Transcribe with Parakeet
            try:
                outputs = self.transcriber.transcribe([temp_path])

                if outputs and len(outputs) > 0:
                    transcript_text = ""

                    # Extract text from output
                    if hasattr(outputs[0], 'text'):
                        transcript_text = outputs[0].text.strip()
                    elif isinstance(outputs[0], str):
                        transcript_text = outputs[0].strip()
                    else:
                        transcript_text = str(outputs[0]).strip()

                    if transcript_text:
                        # Create segment in Whisper-compatible format
                        # Since Parakeet doesn't provide detailed segments, create one
                        segment = type('Segment', (), {
                            'text': transcript_text,
                            'start': 0.0,
                            'end': len(input_sample) / self.sample_rate,
                            'no_speech_prob': 0.0,
                            'tokens': [],
                            'temperature': 0.0,
                            'avg_logprob': 0.0,
                            'compression_ratio': 0.0,
                            'words': None
                        })()
                        result = [segment]

            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

        finally:
            if ServeClientParakeet.SINGLE_MODEL and self.single_model:
                ServeClientParakeet.SINGLE_MODEL_LOCK.release()

        return result

    def handle_transcription_output(self, result, duration):
        """
        Handle the transcription output

        Args:
            result: Transcription results
            duration: Duration of audio chunk
        """
        segments = []
        if len(result):
            self.t_start = None
            last_segment = self.update_segments(result, duration)
            segments = self.prepare_segments(last_segment)

        if len(segments):
            self.send_transcription_to_client(segments)

    def cleanup(self):
        """Clean up resources"""
        super().cleanup()
        # Clean up temp directory
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            import shutil
            try:
                shutil.rmtree(self.temp_dir)
            except:
                pass
