FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir .

COPY bot.py dashboard.py metrics_observer.py server.py ./
COPY static/ static/

RUN mkdir -p data/sessions data/recordings

ENV PORT=8000
EXPOSE $PORT

CMD ["python", "server.py"]
