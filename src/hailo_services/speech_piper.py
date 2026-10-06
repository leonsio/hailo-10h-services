"""CPU Piper adapter and independent bounded speech lifecycle.

Only locally provisioned ONNX/JSON voice pairs are used. A future accelerator
adapter can implement the same start/synthesize/close contract.
"""

import asyncio
import io
import json
import logging
import re
import wave
from concurrent.futures import ThreadPoolExecutor
from math import gcd
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from .errors import BusyError

_LOG = logging.getLogger(__name__)


def _language_matches(expected, actual):
    """Compare language codes with optional regional specificity.

    Args:
        expected (str): Requested or configured language.
        actual (str): Language declared by the voice metadata.

    Returns:
        bool: Whether language and any requested region match.
    """
    expected, actual = expected.replace("-", "_").lower(), actual.replace("-", "_").lower()
    return expected == actual or ("_" not in expected and expected == actual.split("_")[0])


def installed_voices(settings):
    """List provisioned voice IDs and languages without loading models.

    Args:
        settings (Settings): Local directory containing ONNX and JSON pairs.

    Returns:
        list[dict[str, str]]: Voice identifiers and declared languages, without paths.
    """
    voices = []
    directory = Path(settings.piper_voice_dir).resolve()
    for model in sorted(directory.glob("*.onnx")):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", model.stem) or model.resolve().parent != directory:
            continue
        try:
            metadata = json.loads(Path(str(model) + ".json").read_text(encoding="utf-8"))
            language = metadata["language"]["code"]
            if isinstance(language, str):
                voices.append({"id": model.stem, "language": language})
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return voices


class PiperBackend:
    """Own one resident CPU voice and replace it when the selected voice changes."""

    def __init__(self, settings):
        """Store settings without importing optional Piper dependencies.

        Args:
            settings (Settings): Voice selection and inference limits.

        Returns:
            None: Creates an unloaded adapter.
        """
        self.settings = settings
        self.voice = None
        self.loaded_path = None
        self.language = None

    def _path(self, voice=None):
        """Resolve an installed ID or the trusted configured default path.

        Args:
            voice (str | None): Client-selected installed ID.

        Returns:
            Path: Voice file path whose ONNX and JSON files both exist.

        Raises:
            ValueError: Voice is invalid or missing.
        """
        selection = voice or self.settings.piper_voice
        if voice is None and Path(selection).is_absolute():
            path = Path(selection)
        else:
            if not re.fullmatch(r"[A-Za-z0-9_-]+", selection):
                raise ValueError("Piper voice must be an installed voice ID")
            directory = Path(self.settings.piper_voice_dir).resolve()
            path = (directory / f"{selection}.onnx").resolve()
            if path.parent != directory:
                raise ValueError("Piper voice must remain inside the configured directory")
        if not path.is_file() or not Path(str(path) + ".json").is_file():
            raise ValueError(f"Piper voice is not installed: {selection} (ONNX and JSON required)")
        return path

    def _load(self, path, expected):
        """Load a voice on CPU after verifying its declared language.

        Args:
            path (Path): Provisioned voice model path.
            expected (str | None): Expected language, when specified.

        Returns:
            None: Makes this voice resident, dropping the previous voice.

        Raises:
            ValueError: Voice metadata does not match the expected language.
        """
        metadata = json.loads(Path(str(path) + ".json").read_text(encoding="utf-8"))
        language = metadata["language"]["code"]
        if expected and not _language_matches(expected, language):
            raise ValueError(f"Selected Piper voice speaks {language}, not {expected}")
        if path != self.loaded_path:
            from piper import PiperVoice

            self.voice = None
            self.loaded_path = None
            self.voice = PiperVoice.load(str(path), use_cuda=False)
            self.loaded_path = path
        self.language = language

    def start(self):
        """Initialize the configured default voice once.

        Returns:
            None: Keeps a CPU voice resident for subsequent requests.
        """
        self._load(self._path(), self.settings.piper_language)

    def synthesize(self, request):
        """Produce bounded mono audio with the requested speed and local voice.

        Args:
            request (SpeechRequest): Validated synthesis request.

        Returns:
            tuple[bytes, str, int]: Audio bytes, MIME type and sample rate.

        Raises:
            ValueError: Voice/language is invalid or generated audio exceeds its limit.
        """
        from piper import SynthesisConfig

        self._load(self._path(request.voice), request.language)
        buffer = io.BytesIO()
        frames = 0
        rate = self.voice.config.sample_rate
        with wave.open(buffer, "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(rate)
            for chunk in self.voice.synthesize(
                request.input, syn_config=SynthesisConfig(length_scale=1.0 / request.speed)
            ):
                data = chunk.audio_int16_bytes
                frames += len(data) // 2
                if frames > rate * self.settings.max_audio_seconds:
                    raise ValueError("Generated speech exceeds max_audio_seconds; split the text")
                output.writeframesraw(data)
        audio = buffer.getvalue()
        if request.response_format == "pcm":
            with wave.open(io.BytesIO(audio), "rb") as source:
                samples = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2")
            divisor = gcd(rate, 24000)
            samples = resample_poly(samples.astype(np.float32), 24000 // divisor, rate // divisor)
            audio = np.clip(np.rint(samples), -32768, 32767).astype("<i2").tobytes()
            return audio, "audio/pcm", 24000
        return audio, "audio/wav", rate

    def close(self):
        """Release the resident ONNX voice.

        Returns:
            None: Drops CPU model references.
        """
        self.voice = None
        self.loaded_path = None


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
