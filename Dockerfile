# ── Base image ────────────────────────────────────────────────────────
FROM python:3.11-slim

LABEL maintainer="your-email@example.com"
LABEL description="BDD100K object detection data analysis pipeline"

# ── System dependencies ───────────────────────────────────────────────
# libgl1 and libglib2.0-0 are required by Pillow for image decoding
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────
# Copy requirements first so this layer is cached unless deps change
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ──────────────────────────────────────────────────
COPY data_analysis/ ./data_analysis/
COPY tests/         ./tests/

# Create output directory
RUN mkdir -p /app/outputs

# ── Entrypoint ────────────────────────────────────────────────────────
# The BDD100K dataset is mounted at runtime via docker run -v
# Default command runs the full analysis pipeline
CMD ["python", "-m", "data_analysis.main"]
