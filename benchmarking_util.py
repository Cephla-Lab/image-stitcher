import contextlib
import logging
import time
from typing import Generator


@contextlib.contextmanager
def debug_timing(span_name: str) -> Generator[None, None, None]:
    """Log a message to debug level with the time to run the code in this context."""
    start_time = time.time()
    yield
    total_time = time.time() - start_time
    logging.debug(f"{span_name}: {total_time:0.3f}s")


@contextlib.contextmanager
def profile_context(to_file: str) -> Generator[None, None, None]:
    """Run profiling with cProfile within this context and dump the output to the given file."""
    import cProfile

    pr = cProfile.Profile()
    pr.enable()
    yield
    pr.disable()
    pr.dump_stats(to_file)
