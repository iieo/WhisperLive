import threading
import time
import queue
import logging
from typing import List, Tuple
import numpy as np


class BatchManager:
    """
    Manages batching of transcription requests to improve efficiency.
    """

    def __init__(self, batch_processor, batch_size=8, batch_timeout=0.1):
        """
        Initializes the BatchManager.

        Args:
            batch_processor (callable): The function that processes a batch of audio samples.
            batch_size (int): The maximum number of requests to batch together.
            batch_timeout (float): The maximum time to wait before processing a batch, even if it's not full.
        """
        self.batch_processor = batch_processor
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.job_queue = queue.Queue()
        self.processing_thread = threading.Thread(
            target=self._batch_processing_loop, daemon=True)
        self.active = True
        self.processing_thread.start()
        logging.info(
            f"BatchManager initialized with batch_size={batch_size} and batch_timeout={batch_timeout}s.")

    def submit_job(self, audio_sample: np.ndarray) -> List:
        """
        Submits an audio sample for transcription and waits for the result.

        Args:
            audio_sample (np.ndarray): The audio data to be transcribed.

        Returns:
            list: The transcription result for the submitted audio sample.
        """
        result_queue = queue.Queue()
        self.job_queue.put((audio_sample, result_queue))
        try:
            # Wait for the result to be placed in the queue
            result = result_queue.get()
            return result
        except Exception as e:
            logging.error(f"Error retrieving result from job: {e}")
            return []

    def _batch_processing_loop(self):
        """
        The main loop that collects jobs, processes them in a batch, and distributes results.
        """
        while self.active:
            batch: List[Tuple[np.ndarray, queue.Queue]] = []
            start_time = time.time()

            # Collect jobs for the batch
            while len(batch) < self.batch_size and (time.time() - start_time) < self.batch_timeout:
                try:
                    # Wait for a job, but with a timeout to respect the batch_timeout
                    timeout = self.batch_timeout - (time.time() - start_time)
                    if timeout <= 0:
                        break
                    job = self.job_queue.get(timeout=timeout)
                    if job is None:  # Sentinel for stopping the thread
                        continue
                    batch.append(job)
                except queue.Empty:
                    # If the queue is empty, break the inner loop to process the current batch
                    break

            if not batch:
                continue

            # Prepare batch for transcription
            input_samples = [job[0] for job in batch]
            result_queues = [job[1] for job in batch]

            try:
                # Process the batch
                batch_results = self.batch_processor(input_samples)

                # Distribute results
                if batch_results and len(batch_results) == len(result_queues):
                    for result, res_queue in zip(batch_results, result_queues):
                        res_queue.put(result)
                else:
                    # If transcription fails or returns unexpected results, send empty results
                    logging.error(
                        f"Batch transcription returned {len(batch_results)} results for a batch of size {len(result_queues)}.")
                    for res_queue in result_queues:
                        res_queue.put([])

            except Exception as e:
                logging.error(f"Error during batch transcription: {e}")
                # Notify all clients in the batch about the failure
                for res_queue in result_queues:
                    res_queue.put([])

    def stop(self):
        """
        Stops the batch processing thread.
        """
        self.active = False
        # Put a dummy item to unblock the queue.get()
        self.job_queue.put(None)
        self.processing_thread.join(timeout=2.0)
        logging.info("BatchManager stopped.")
