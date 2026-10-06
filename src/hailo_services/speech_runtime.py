"""Independent bounded scheduling and lifecycle for speech synthesis adapters."""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

from .errors import BusyError
from .speech_piper import PiperBackend

_LOG = logging.getLogger(__name__)


class SpeechRuntime:
    """Schedule CPU TTS independently from Hailo and LiteRT owner threads."""

    def __init__(self, settings, backend=None):
        """Create a bounded owner executor for an optional speech adapter.

        Args:
            settings (Settings): Enablement, queue size and inference deadline.
            backend (PiperBackend | None): Injectable implementation of the speech contract.

        Returns:
            None: Creates an unstarted runtime.
        """
        self.settings = settings
        self.backend = backend if backend is not None else PiperBackend(settings)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="piper-owner")
        self.ready = False
        self.pending = 0
        self.error = None

    async def start(self):
        """Load Piper when enabled, keeping other services available on failure.

        Returns:
            None: Records readiness or a diagnostic startup error.
        """
        if self.settings.piper_enabled:
            try:
                await asyncio.get_running_loop().run_in_executor(self.executor, self.backend.start)
                self.ready = True
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"
                _LOG.exception("Piper startup failed; other backends remain available")

    def _completed(self, future):
        """Release capacity only after native synthesis actually completes.

        Args:
            future (asyncio.Future): Completed owner-thread inference.

        Returns:
            None: Observes late exceptions and releases one queue slot.
        """
        self.pending -= 1
        if not future.cancelled():
            future.exception()

    async def synthesize(self, request):
        """Enforce configured input limits and await shielded CPU synthesis.

        Args:
            request (SpeechRequest): Validated API request.

        Returns:
            tuple[bytes, str, int]: Generated audio, MIME type and sample rate.

        Raises:
            BusyError: Piper is disabled/unavailable or the independent queue is full.
            ValueError: Text exceeds the configured input length.
            asyncio.TimeoutError: Client deadline expires while native work continues.
        """
        if len(request.input) > self.settings.piper_max_input_chars:
            raise ValueError(
                f"Speech input exceeds {self.settings.piper_max_input_chars} characters"
            )
        if not self.ready:
            raise BusyError(f"Piper is unavailable: {self.error or 'disabled'}")
        if self.pending >= self.settings.queue_size:
            raise BusyError("Piper inference queue is full")
        self.pending += 1
        future = asyncio.get_running_loop().run_in_executor(
            self.executor, self.backend.synthesize, request
        )
        future.add_done_callback(self._completed)
        return await asyncio.wait_for(asyncio.shield(future), self.settings.request_timeout)

    def status(self):
        """Expose speech configuration and independent readiness.

        Returns:
            dict: CPU TTS status without client text or credentials.
        """
        return {
            "enabled": self.settings.piper_enabled,
            "ready": self.ready,
            "model": "piper",
            "device": "cpu",
            "voice": self.settings.piper_voice,
            "language": self.settings.piper_language,
            "pending": self.pending,
            "error": self.error,
        }

    async def close(self):
        """Drain native work before releasing the resident voice and executor.

        Returns:
            None: Stops accepting requests and closes owned CPU resources.
        """
        self.ready = False
        try:
            await asyncio.get_running_loop().run_in_executor(self.executor, self.backend.close)
        finally:
            self.executor.shutdown(wait=True, cancel_futures=True)
