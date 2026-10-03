import logging

import uvicorn

from .app import create_app
from .config import Settings


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    settings = Settings.from_env()
    if settings.debug_log:
        logging.getLogger("hailo_services").setLevel(logging.DEBUG)
    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        workers=1,
        ws_max_size=settings.max_body,
        timeout_graceful_shutdown=30,
    )


if __name__ == "__main__":
    main()
