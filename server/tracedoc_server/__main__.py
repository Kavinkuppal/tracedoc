"""Run the server without logging or parsing client addresses.

    TRACEDOC_DB=postgresql://... python -m tracedoc_server [--host 127.0.0.1] [--port 8000]

Production must terminate TLS in front of this process. `--allow-http` exists only for
local testing.
"""

from __future__ import annotations

import argparse
import os

import uvicorn

from .app import create_app
from .db import Store

# No access logger at all: uvicorn's access format includes the client address.
LOG_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {"plain": {"format": "%(asctime)s %(levelname)s %(name)s: %(message)s"}},
    "handlers": {"stderr": {"class": "logging.StreamHandler", "formatter": "plain"}},
    "loggers": {
        "uvicorn": {"handlers": ["stderr"], "level": "INFO", "propagate": False},
        "uvicorn.error": {"handlers": ["stderr"], "level": "INFO", "propagate": False},
        "uvicorn.access": {"handlers": [], "level": "CRITICAL", "propagate": False},
        "tracedoc_server": {"handlers": ["stderr"], "level": "INFO", "propagate": False},
    },
}


def server_config(app, host: str, port: int) -> uvicorn.Config:  # noqa: ANN001
    return uvicorn.Config(
        app,
        host=host,
        port=port,
        access_log=False,
        proxy_headers=False,  # never parse X-Forwarded-For
        server_header=False,
        date_header=False,
        log_config=LOG_CONFIG,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--allow-http", action="store_true", help="Local testing only.")
    args = parser.parse_args()
    store = Store(os.environ["TRACEDOC_DB"])
    store.migrate()
    app = create_app(store, allow_http=args.allow_http)
    uvicorn.Server(server_config(app, args.host, args.port)).run()


if __name__ == "__main__":
    main()
