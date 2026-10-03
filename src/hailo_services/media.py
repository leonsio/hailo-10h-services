import base64
import io
import math

import numpy as np
import soundfile as sf
from PIL import Image, UnidentifiedImageError
from scipy.signal import resample_poly

_MAX_IMAGE_PIXELS = 50_000_000
Image.MAX_IMAGE_PIXELS = _MAX_IMAGE_PIXELS


def decode_base64(value: str, limit: int) -> bytes:
    if value.startswith("data:"):
        header, value = value.split(",", 1)
        if ";base64" not in header:
            raise ValueError("Only base64 data URLs are supported")
    if len(value) > ((limit + 2) // 3) * 4:
        raise ValueError("Media exceeds size limit")
    try:
        data = base64.b64decode(value, validate=True)
    except (ValueError, base64.binascii.Error) as exc:
        raise ValueError("Invalid base64 media") from exc
    if len(data) > limit:
        raise ValueError("Media exceeds size limit")
    return data


def image_frame(value: str, limit: int) -> np.ndarray:
    # URL fetching is deliberately excluded: callers supply their snapshot bytes.
    try:
        with Image.open(io.BytesIO(decode_base64(value, limit))) as image:
            if image.width * image.height > _MAX_IMAGE_PIXELS:
                raise ValueError("Image has too many pixels")
            # Let JPEG downsample in the decoder before allocating the full-resolution
            # RGB buffer. Phone photos can be 48 MP, while the VLM input is only 336².
            image.draft("RGB", (672, 672))
            # Pillow's buffer-backed ndarray is read-only; Hailo GenAI requires
            # writable, contiguous frame storage at the native API boundary.
            return np.array(image.convert("RGB").resize((336, 336)), dtype=np.uint8, copy=True)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("Invalid or oversized image") from exc


def normalize_audio(samples: np.ndarray, rate: int, max_seconds: int) -> np.ndarray:
    if rate < 8000 or rate > 192000:
        raise ValueError("Audio rate must be 8000..192000 Hz")
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    if samples.ndim != 1 or not samples.size:
        raise ValueError("Audio must contain samples")
    if samples.size > rate * max_seconds:
        raise ValueError("Audio exceeds duration limit")
    if not np.isfinite(samples).all():
        raise ValueError("Audio contains non-finite samples")
    if rate != 16000:
        divisor = math.gcd(rate, 16000)
        samples = resample_poly(samples, 16000 // divisor, rate // divisor)
    return np.ascontiguousarray(np.clip(samples, -1, 1), dtype="<f4")


def audio_file(data: bytes, max_seconds: int) -> np.ndarray:
    try:
        with sf.SoundFile(io.BytesIO(data)) as audio:
            if audio.frames > audio.samplerate * max_seconds:
                raise ValueError("Audio exceeds duration limit")
            if audio.channels > 8:
                raise ValueError("Too many audio channels")
            samples = audio.read(dtype="float32", always_2d=True)
            return normalize_audio(samples, audio.samplerate, max_seconds)
    except (sf.LibsndfileError, RuntimeError) as exc:
        raise ValueError("Unsupported audio; use WAV, FLAC or OGG supported by libsndfile") from exc


def audio_metadata(data: bytes) -> dict:
    """Read decoder-independent audio facts for logs without exposing audio content."""
    try:
        info = sf.info(io.BytesIO(data))
    except (sf.LibsndfileError, RuntimeError) as exc:
        raise ValueError("Unsupported audio; use WAV, FLAC or OGG supported by libsndfile") from exc
    return {
        "container": info.format,
        "codec": info.subtype,
        "sample_rate_hz": info.samplerate,
        "channels": info.channels,
        "frames": info.frames,
        "duration_seconds": info.duration,
    }
