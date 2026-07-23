# Single-stage Python image for the research-paper-extractor pipeline.
#
# Runtime deps only — no test tooling, no build cache leakage. Data,
# credentials, and logs are provided at runtime via volume mounts and
# --env-file, never baked into the image.

FROM python:3.11-slim

WORKDIR /app

# build-essential covers any C-extension deps that show up transitively.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency manifests first so the pip layer caches across code edits.
COPY requirements.txt setup.py ./
COPY src/ ./src/

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -e .

# Ship configs with the image — small, versioned, non-secret.
COPY configs/ ./configs/

# data/ + logs/ are runtime volumes; create empty targets so the container
# can start even without a mount.
RUN mkdir -p data/raw data/processed data/extracted logs

ENV PYTHONUNBUFFERED=1

# Use the pipeline CLI as the entrypoint so `docker run <image> --provider ...`
# reads naturally. CMD is a safe default (prints --help).
ENTRYPOINT ["python", "-m", "rpextractor.pipeline.main"]
CMD ["--help"]
