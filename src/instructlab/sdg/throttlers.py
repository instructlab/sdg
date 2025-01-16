import threading

DEFAULT_TOLERANCE = 0.2  # Fraction to reduce workers on failure

class AdaptiveThrottler:
    def __init__(self, min_workers, max_workers, initial_workers, tolerance=DEFAULT_TOLERANCE):
        self.min_workers = min_workers  # Lower limit of workers
        self.max_workers = max_workers  # Upper limit of workers, same as num_cpus cli argument
        self.current_workers = initial_workers  # Start with this number
        self.tolerance = tolerance  # Reduce workers by this fraction on error
        self.lock = threading.Lock()  # Ensure thread-safe updates

    def adjust_workers(self, success=True):
        """Adjust the number of workers based on success or failure."""
        with self.lock:  # Use a lock to avoid race conditions in multi-threading
            if success:
                # Gradually increase workers up to max_workers
                if self.current_workers < self.max_workers:
                    self.current_workers += 1
            else:
                # Reduce workers by a fraction on failure, respecting min_workers
                if self.current_workers > self.min_workers:
                    self.current_workers = max(
                        self.min_workers,
                        int(self.current_workers * (1 - self.tolerance)),
                    )

    def get_workers(self):
        """Get the current number of workers."""
        with self.lock:
            return self.current_workers
