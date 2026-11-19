# One image, two roles: the API (default command) and the Streamlit UI (overridden in docker-compose).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install dependencies first so Docker can cache this layer when only code changes.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the application code (data/ and .env are excluded by .dockerignore).
COPY app ./app
COPY scripts ./scripts
COPY evals ./evals
COPY ui ./ui
COPY pyproject.toml .

# Run as a non-root user (security best practice).
RUN useradd --create-home appuser && mkdir -p /app/data && chown -R appuser /app
USER appuser

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)" || exit 1

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
