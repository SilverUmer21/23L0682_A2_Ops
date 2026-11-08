"""Run the app with Flask's own server, debug mode OFF:

    python -m webapp          # web UI
    python -m webapp api      # model API

Debug mode stays off because it exposes an interactive debugger that can run arbitrary code in the server process
and reloads on every file change. Inside Docker the app is started with gunicorn instead (see the Dockerfile).
"""
import os
import sys

from webapp import create_app
from webapp.api_app import create_api_app

if __name__ == "__main__":
    app = create_api_app() if sys.argv[1:] == ["api"] else create_app()
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "8000")), debug=False)
