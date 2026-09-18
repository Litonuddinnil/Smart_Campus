"""Root entrypoint for the GridWise API.

Local dev:               python app.py
Production (Docker SDK): gunicorn --bind 0.0.0.0:7860 app:app
                          (the Dockerfile's CMD points at this same `app:app`)
Hugging Face Gradio SDK:  runs `python app.py` directly -- see the "Note on
                          the dev server under Gradio SDK" in README.md for
                          why threaded=True matters here specifically.

All actual logic lives in the gridwise package (see gridwise/api/app.py for
the route definitions). This file only wires up the WSGI object and the dev
server invocation.
"""
from gridwise import config
from gridwise.api import create_app

app = create_app()

if __name__ == "__main__":
    # threaded=True: without it, Flask's dev server handles one request at a
    # time. A Docker-SDK deployment doesn't need this (gunicorn handles
    # concurrency instead, see the Dockerfile), but a Gradio-SDK deployment
    # has no gunicorn in front of this process, so this is what lets a
    # second judge request proceed while an LLM call is in flight on the
    # first.
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG, threaded=True)
