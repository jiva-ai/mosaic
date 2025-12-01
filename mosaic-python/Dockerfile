# Use Python 3.11 as base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Label for GitHub Container Registry linking
LABEL org.opencontainers.image.source="https://github.com/jiva-ai/mosaic"

# Install system dependencies required for building Python packages
# and optional GPU tools (these are optional but included for completeness)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package installer)
RUN pip install --no-cache-dir uv

# Copy dependency files first for better layer caching
COPY pyproject.toml uv.lock .python-version ./
# Copy README.md (required by pyproject.toml, even if empty)
COPY README.md ./

# Install dependencies using uv
# This will create a virtual environment and install all dependencies
RUN uv sync --frozen --no-dev

# Copy the project source code
COPY mosaic/ ./mosaic/
COPY mosaic_comms/ ./mosaic_comms/
COPY mosaic_config/ ./mosaic_config/
COPY mosaic_model_runtime/ ./mosaic_model_runtime/
COPY mosaic_planner/ ./mosaic_planner/
COPY mosaic_stats/ ./mosaic_stats/

# Activate the virtual environment and add it to PATH
ENV PATH="/app/.venv/bin:$PATH"

# Set Python path to include the project root
ENV PYTHONPATH="/app:$PYTHONPATH"

# Ephemeral port range (49152-65535) for dynamic connections between nodes
EXPOSE 49152-65535/tcp 49152-65535/udp

# Default command (can be overridden)
CMD ["python", "-m", "mosaic.mosaic"]

