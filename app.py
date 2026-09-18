"""Root entrypoint for the GridWise API.

Local dev:     python app.py
Production:    gunicorn --bind 0.0.0.0:7860 app:app
Hugging Face Spaces (Docker SDK) runs the CMD from the Dockerfile, which
points at this same `app:app` object.

All actual logic lives in the gridwise package (see gridwise/api/app.py for
the route definitions and gridwise/README.md-referenced modules for the
pipeline). This file only wires up the WSGI object and the dev server.
"""
from gridwise import config
from gridwise.api import create_app

app = create_app()

if __name__ == "__main__":
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)
