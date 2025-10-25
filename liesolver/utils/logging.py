import io
import logging
import sys
from contextlib import contextmanager
from typing import Generator

import timeit
from functools import wraps


def timing(f):
    """Decorator for measuring the execution time of methods."""

    @wraps(f)
    def wrapper(*args, **kwargs):
        ts = timeit.default_timer()
        result = f(*args, **kwargs)
        te = timeit.default_timer()
        verbose = kwargs.get('verbose', 1)
        if verbose > 0:
            print(f"{f.__name__!r} took {te - ts:.2f} s")
            sys.stdout.flush()
        return result

    return wrapper



class StreamToLogger(io.StringIO):
    """
    Redirects a stream (stdout or stderr) to a logger.
    """

    def __init__(self, logger: logging.Logger, log_level: int = logging.INFO) -> None:
        """
        Initialize the StreamToLogger.

        Args:
            logger (logging.Logger): The logger instance to write messages to.
            log_level (int, optional): The logging level. Defaults to logging.INFO.
        """
        super().__init__()
        self.logger = logger
        self.log_level = log_level

    def write(self, message: str) -> None:
        """
        Write a message to the logger.

        Args:
            message (str): The message to write.
        """
        if message.strip():
            self.logger.log(self.log_level, message.strip())

    def flush(self) -> None:
        """
        Flush the stream. This is a no-op for StreamToLogger.
        """
        pass


@contextmanager
def log_output() -> Generator[None, None, None]:
    """
    Context manager to redirect stdout and stderr to a logger.
    """
    logger = logging.getLogger(__name__)
    stdout_logger = StreamToLogger(logger, logging.INFO)
    stderr_logger = StreamToLogger(logger, logging.ERROR)

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    sys.stdout = stdout_logger
    sys.stderr = stderr_logger

    try:
        yield
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
