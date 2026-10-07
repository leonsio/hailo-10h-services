"""Command-line entry point starting one uvicorn service worker."""

import logging

import uvicorn

from .app import create_app
from .config import Settings
from .logging_utils import install_inline_data_redaction


def main():
    """Start the command-line entry point for this module.

    Returns:
        None: Runs the configured command until completion.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    install_inline_data_redaction()
    settings = Settings.from_env()
    if settings.debug_log:
        logging.getLogger("hailo_services").setLevel(logging.DEBUG)
    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        workers=1,
        proxy_headers=False,  # MCP network access uses the actual socket peer.
        ws_max_size=settings.max_body,
        timeout_graceful_shutdown=30,
    )


if __name__ == "__main__":
    main()
