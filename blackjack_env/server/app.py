# SPDX-License-Identifier: BSD-3-Clause

"""FastAPI application for the Blackjack Environment.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    uvicorn server.app:app --host 0.0.0.0 --port 8000
    python -m server.app --port 8000
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with 'pip install -e .'"
    ) from e

try:
    from ..models import BlackjackAction, BlackjackObservation
    from .blackjack_env_environment import BlackjackEnvironment
except ImportError:
    from models import BlackjackAction, BlackjackObservation
    from server.blackjack_env_environment import BlackjackEnvironment


app = create_app(
    BlackjackEnvironment,
    BlackjackAction,
    BlackjackObservation,
    env_name="blackjack_env",
    max_concurrent_envs=4,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(host=args.host, port=args.port)
