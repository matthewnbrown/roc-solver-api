# Use Python slim image for smaller base
FROM python:3.9-slim

# Build argument to choose between CPU and GPU versions
ARG REQUIREMENTS_FILE=requirements-cpu.txt

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

COPY ${REQUIREMENTS_FILE} requirements.txt

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip cache purge

COPY . .

# Create a non-root user and set ownership
RUN groupadd -r appuser && useradd -r -g appuser -d /app appuser \
    && mkdir -p model data \
    && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "run.py"]
