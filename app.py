"""Root entrypoint for the GridWise API.

Local dev:               python app.py
Production / Hugging Face Space (Docker SDK): gunicorn --bind 0.0.0.0:7860
                          app:app  (the Dockerfile's CMD points at this same
                          `app:app`)

All actual logic lives in the gridwise package (see gridwise/api/app.py for
the route definitions). This file only wires up the WSGI object and the
local dev-server invocation.
"""
from gridwise import config
from gridwise.api import create_app

app = create_app()

if __name__ == "__main__":
    # Local development only. The Dockerfile (and therefore the deployed HF
    # Space) runs gunicorn instead, which handles concurrency in its own
    # worker/thread pool. threaded=True is kept here so a second `curl`
    # against a locally-running `python app.py` doesn't block behind a slow
    # LLM call.
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG, threaded=True)
