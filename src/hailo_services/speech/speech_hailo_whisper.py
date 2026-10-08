"""Hailo Whisper model creation and transcription on the shared device owner."""

from hailo_services.shared.errors import BusyError


def create_model(device, path):
    """Create resident Whisper on the existing SHARED VDevice.

    Args:
        device (VDevice): Device owned and scheduled by the Hailo backend.
        path (str): Provisioned Whisper HEF file.

    Returns:
        Speech2Text: Native speech model; the Hailo backend owns its release.
    """
    from hailo_platform.genai import Speech2Text

    return Speech2Text(device, path)


def transcribe(model, audio, language, timeout):
    """Convert normalized mono audio into text with the resident model.

    Args:
        model (Speech2Text | None): Resident native model or None when disabled.
        audio (np.ndarray): Normalized 16 kHz mono float32 samples.
        language (str): Requested transcription language.
        timeout (float): Native inference deadline in seconds.

    Returns:
        str: Joined, trimmed transcription segments.

    Raises:
        BusyError: Whisper is disabled or unavailable.
    """
    if model is None:
        raise BusyError("Whisper is disabled")
    from hailo_platform.genai import Speech2TextTask

    segments = model.generate_all_segments(
        audio_data=audio,
        task=Speech2TextTask.TRANSCRIBE,
        language=language,
        timeout_ms=int(timeout * 1000),
    )
    return "".join(segment.text for segment in segments).strip()
