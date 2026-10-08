"""CPU Piper adapter, voice discovery and audio encoding.

Only locally provisioned ONNX/JSON voice pairs are used. A future accelerator
adapter can implement the same start/synthesize/close contract.
"""

import io
import json
import logging
import re
import time
import wave
from importlib.metadata import PackageNotFoundError, version
from math import gcd
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from hailo_services.speech.speech_common import language_matches

_LOG = logging.getLogger(__name__)


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
        self.default_language = None
        self.version = "unknown"

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
        if expected and not language_matches(expected, language):
            raise ValueError(f"Selected Piper voice speaks {language}, not {expected}")
        if path != self.loaded_path:
            from piper import PiperVoice

            started = time.perf_counter()
            _LOG.info(
                "Loading Piper voice device=cpu voice=%s language=%s model_path=%s",
                path.stem,
                language,
                path,
            )
            self.voice = None
            self.loaded_path = None
            self.voice = PiperVoice.load(str(path), use_cuda=False)
            self.loaded_path = path
            _LOG.info(
                "Piper voice loaded version=%s device=cpu voice=%s language=%s sample_rate_hz=%d load_ms=%.1f",
                self.version,
                path.stem,
                language,
                self.voice.config.sample_rate,
                (time.perf_counter() - started) * 1000,
            )
        self.language = language

    def start(self):
        """Initialize the configured default voice once.

        Returns:
            None: Keeps a CPU voice resident for subsequent requests.
        """
        try:
            self.version = version("piper-tts")
        except PackageNotFoundError:
            self.version = "unknown"
        _LOG.info(
            "Starting Piper TTS version=%s device=cpu default_voice=%s configured_language=%s",
            self.version,
            self.settings.piper_voice,
            self.settings.piper_language,
        )
        self._load(self._path(), self.settings.piper_language)
        self.default_language = self.language

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
