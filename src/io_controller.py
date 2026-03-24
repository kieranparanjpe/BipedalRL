import sys
import threading
import queue
from typing import Optional

class IOController:
    def __init__(self) -> None:
        self._inbox: queue.Queue[str] = queue.Queue()
        self._stop_event = threading.Event()
        self._reader_thread = threading.Thread(
            target=self._listen_stdin,
            daemon=True,
        )
        self._reader_thread.start()

    def _listen_stdin(self) -> None:
        try:
            for line in sys.stdin:
                if self._stop_event.is_set():
                    break
                self._inbox.put(line.rstrip("\n"))
        finally:
            self._stop_event.set()

    def write(self, message: str) -> None:
        print(message, flush=True)

    def read(self, block: bool = False, timeout: Optional[float] = None) -> Optional[str]:
        try:
            return self._inbox.get(block=block, timeout=timeout)
        except queue.Empty:
            return None

    def has_message(self) -> bool:
        return not self._inbox.empty()

    def stop(self) -> None:
        self._stop_event.set()

    def closed(self) -> bool:
        return self._stop_event.is_set() and self._inbox.empty()