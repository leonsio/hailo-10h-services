"""Command-line entry point starting the gateway and resident worker processes."""

import logging
import os

import uvicorn

from .app import create_app
from .config import Settings
from .logging_utils import install_inline_data_redaction
from .process_app import create_process_app


def _process_mode_enabled() -> bool:
    """Return whether production inference should use isolated child processes."""
    return os.getenv("HAILO_PROCESS_MODE", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def main():
    """Start the command-line entry point for this module.

    Returns:
        None: Runs the configured command until completion.

    Notes:
        ``HAILO_PROCESS_MODE=0`` keeps the previous owner-thread architecture as
        an emergency compatibility fallback without changing public APIs.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    install_inline_data_redaction()
    settings = Settings.from_env()
    if settings.debug_log:
        logging.getLogger("hailo_services").setLevel(logging.DEBUG)
    process_mode = _process_mode_enabled()
    logging.getLogger(__name__).info(
        "Execution mode=%s cpu_count=%s",
        "process-isolated" if process_mode else "legacy-threaded",
        os.cpu_count(),
    )
    application = create_process_app(settings) if process_mode else create_app(settings)
    uvicorn.run(
        application,
        host=settings.host,
        port=settings.port,
        workers=1,
        proxy_headers=False,  # MCP network access uses the actual socket peer.
        ws_max_size=settings.max_body,
        timeout_graceful_shutdown=30,
    )


if __name__ == "__main__":
    main()
