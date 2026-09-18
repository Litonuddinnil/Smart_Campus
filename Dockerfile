FROM python:3.12-slim

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code (the gridwise package + root entrypoint).
COPY app.py .
COPY gridwise ./gridwise

# 7860 is the Hugging Face Spaces default container port; also documented
# in README.md and used as the default in gridwise/config.py, so the
# Dockerfile, the Space, and the local docs all agree. It's only a default,
# though -- see CMD below.
EXPOSE 7860
ENV PORT=7860
ENV HOST=0.0.0.0

# No secrets are baked into the image. LLM_API_KEY (or the provider's own
# env var name, e.g. GROQ_API_KEY) must be supplied at container-start time:
# `docker run -e ...` locally, Space Secrets on Hugging Face, or the
# platform's environment-variable settings on Render/Fly/Railway/etc.

# gunicorn is a production-grade WSGI server. One worker keeps memory low
# enough for a free CPU-only host; --threads lets it handle a few concurrent
# judge requests without adding process overhead. --timeout is set above the
# 30s per-request judge limit so gunicorn never kills a request the judge
# would still accept.
#
# Shell form (not exec/JSON-array form) so $PORT is substituted at container
# start: platforms like Render assign their own port via a PORT env var and
# expect the process to bind to it, while a plain `docker run` (no PORT set)
# falls back to 7860 via the ENV default above.
CMD gunicorn --bind 0.0.0.0:${PORT:-7860} --workers 1 --threads 4 --timeout 35 app:app
