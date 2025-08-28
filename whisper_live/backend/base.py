import json
import logging
import threading
import time
import queue
import numpy as np


class ServeClientBase(object):
    RATE = 16000
    SERVER_READY = "SERVER_READY"
    DISCONNECT = "DISCONNECT"

    client_uid: str
    """A unique identifier for the client."""
    websocket: object
    """The WebSocket connection for the client."""
    send_last_n_segments: int
    """Number of most recent segments to send to the client."""
    no_speech_thresh: float
    """Segments with no speech probability above this threshold will be discarded."""
    clip_audio: bool
    """Whether to clip audio with no valid segments."""
    same_output_threshold: int
    """Number of repeated outputs before considering it as a valid segment."""

    def __init__(
        self,
        client_uid,
        websocket,
        send_last_n_segments=10,
        no_speech_thresh=0.45,
        clip_audio=False,
        same_output_threshold=10,
        translation_queue=None,
    ):
        self.client_uid = client_uid
        self.websocket = websocket
        self.send_last_n_segments = send_last_n_segments
        self.no_speech_thresh = no_speech_thresh
        self.clip_audio = clip_audio
        self.same_output_threshold = same_output_threshold

        self.timestamp_offset = 0.0
        self.frames_np = None
        self.frames_offset = 0.0
        self.text = []
        self.current_out = ""
        self.prev_out = ""
        self.exit = False
        self.same_output_count = 0
        self.transcript = []
        self.end_time_for_same_output = None
        self.translation_queue = translation_queue

        # threading
        self.lock = threading.Lock()

    def speech_to_text(self):
        """
        Process an audio stream in an infinite loop, continuously transcribing the speech.
        """
        while True:
            if self.exit:
                logging.info("Exiting speech to text thread")
                break

            if self.frames_np is None:
                time.sleep(0.05)
                continue

            if self.clip_audio:
                self.clip_audio_if_no_valid_segment()

            input_bytes, duration = self.get_audio_chunk_for_processing()
            if duration < 1.0:
                time.sleep(0.1)     # wait for audio chunks to arrive
                continue
            try:
                input_sample = input_bytes.copy()
                result = self.transcribe_audio(input_sample)

                if result is None:
                    self.timestamp_offset += duration
                    # wait for voice activity, result is None when no voice activity
                    time.sleep(0.25)
                    continue
                self.handle_transcription_output(result, duration)

            except Exception as e:
                logging.error(
                    f"[ERROR]: Failed to transcribe audio chunk: {e}")
                time.sleep(0.01)

    def transcribe_audio(self, input_sample):
        raise NotImplementedError

    def handle_transcription_output(self, result, duration):
        raise NotImplementedError

    def format_segment(self, start, end, text, completed=False):
        """
        Formats a transcription segment.
        """
        return {
            'start': "{:.3f}".format(start),
            'end': "{:.3f}".format(end),
            'text': text,
            'completed': completed
        }

    def add_frames(self, frame_np):
        """
        Add audio frames (as a numpy array) to the internal buffer.
        This method is thread-safe.
        """
        with self.lock:
            if self.frames_np is not None and self.frames_np.shape[0] > 45*self.RATE:
                self.frames_offset += 30.0
                self.frames_np = self.frames_np[int(30*self.RATE):]
                if self.timestamp_offset < self.frames_offset:
                    self.timestamp_offset = self.frames_offset

            if self.frames_np is None:
                self.frames_np = frame_np.copy()
            else:
                self.frames_np = np.concatenate(
                    (self.frames_np, frame_np), axis=0)

    def clip_audio_if_no_valid_segment(self):
        """
        Clip audio if the current chunk exceeds a duration threshold without a valid segment.
        """
        with self.lock:
            if self.frames_np[int((self.timestamp_offset - self.frames_offset)*self.RATE):].shape[0] > 25 * self.RATE:
                duration = self.frames_np.shape[0] / self.RATE
                self.timestamp_offset = self.frames_offset + duration - 5

    def get_audio_chunk_for_processing(self):
        """
        Retrieves the next chunk of audio data for processing.
        """
        with self.lock:
            samples_take = max(0, (self.timestamp_offset -
                               self.frames_offset) * self.RATE)
            input_bytes = self.frames_np[int(samples_take):].copy()
        duration = input_bytes.shape[0] / self.RATE
        return input_bytes, duration

    def prepare_segments(self, last_segment=None):
        """
        Prepares the segments of transcribed text to be sent to the client.
        """
        segments = []
        if len(self.transcript) >= self.send_last_n_segments:
            segments = self.transcript[-self.send_last_n_segments:].copy()
        else:
            segments = self.transcript.copy()
        if last_segment is not None:
            segments = segments + [last_segment]
        return segments

    def send_transcription_to_client(self, segments):
        """
        Sends the specified transcription segments to the client over the websocket connection.
        """
        try:
            self.websocket.send(
                json.dumps({
                    "uid": self.client_uid,
                    "segments": segments,
                })
            )
        except Exception as e:
            logging.error(f"[ERROR]: Sending data to client: {e}")

    def disconnect(self):
        """
        Notify the client of disconnection.
        """
        try:
            self.websocket.send(json.dumps({
                "uid": self.client_uid,
                "message": self.DISCONNECT
            }))
        except ConnectionClosed:
            logging.info("Client already disconnected.")

    def cleanup(self):
        """
        Perform cleanup tasks.
        """
        logging.info(f"Cleaning up resources for client {self.client_uid}")
        self.exit = True

    def get_segment_no_speech_prob(self, segment):
        return getattr(segment, "no_speech_prob", 0)

    def get_segment_start(self, segment):
        return getattr(segment, "start", 0)

    def get_segment_end(self, segment):
        return getattr(segment, "end", 0)

    def update_segments(self, segments, duration):
        """
        Processes the segments from the ASR model and updates the transcript.
        """
        offset = None
        self.current_out = ''
        last_segment = None

        if len(segments) > 1 and self.get_segment_no_speech_prob(segments[-1]) <= self.no_speech_thresh:
            for s in segments[:-1]:
                text_ = s.text
                self.text.append(text_)
                with self.lock:
                    start = self.timestamp_offset + self.get_segment_start(s)
                    end = self.timestamp_offset + \
                        min(duration, self.get_segment_end(s))
                if start >= end or self.get_segment_no_speech_prob(s) > self.no_speech_thresh:
                    continue

                completed_segment = self.format_segment(
                    start, end, text_, completed=True)
                self.transcript.append(completed_segment)

                if self.translation_queue:
                    try:
                        self.translation_queue.put(
                            completed_segment.copy(), timeout=0.1)
                    except queue.Full:
                        logging.warning(
                            "Translation queue is full, skipping segment")
                offset = min(duration, self.get_segment_end(s))

        if self.get_segment_no_speech_prob(segments[-1]) <= self.no_speech_thresh:
            self.current_out += segments[-1].text
            with self.lock:
                last_segment = self.format_segment(
                    self.timestamp_offset +
                    self.get_segment_start(segments[-1]),
                    self.timestamp_offset +
                    min(duration, self.get_segment_end(segments[-1])),
                    self.current_out,
                    completed=False
                )

        if self.current_out.strip() == self.prev_out.strip() and self.current_out != '':
            self.same_output_count += 1
            if self.end_time_for_same_output is None:
                self.end_time_for_same_output = self.get_segment_end(
                    segments[-1])
        else:
            self.same_output_count = 0
            self.end_time_for_same_output = None

        if self.same_output_count > self.same_output_threshold:
            if not self.text or self.text[-1].strip().lower() != self.current_out.strip().lower():
                self.text.append(self.current_out)
                with self.lock:
                    completed_segment = self.format_segment(
                        self.timestamp_offset,
                        self.timestamp_offset +
                        min(duration, self.end_time_for_same_output),
                        self.current_out,
                        completed=True
                    )
                    self.transcript.append(completed_segment)

                    if self.translation_queue:
                        try:
                            self.translation_queue.put(
                                completed_segment.copy(), timeout=0.1)
                        except queue.Full:
                            logging.warning(
                                "Translation queue is full, skipping segment")

            self.current_out = ''
            offset = min(duration, self.end_time_for_same_output)
            self.same_output_count = 0
            last_segment = None
            self.end_time_for_same_output = None
        else:
            self.prev_out = self.current_out

        if offset is not None:
            with self.lock:
                self.timestamp_offset += offset

        return last_segment
