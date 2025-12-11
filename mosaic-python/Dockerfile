# Use Python 3.11 as base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Label for GitHub Container Registry linking
LABEL org.opencontainers.image.source="https://github.com/jiva-ai/mosaic"

# Install system dependencies required for building Python packages
# Includes dependencies for PyTorch, ONNX, and ONNX Runtime
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
# This will create a virtual environment and install all dependencies including:
# - PyTorch (>=2.0.0) for training and model operations
# - ONNX (>=1.15.0) for model format handling
# - ONNX Runtime (>=1.16.0) for inference execution
# - torchvision, transformers, torch-geometric for various model types
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
# Fix: Use ${PYTHONPATH:-} to avoid warning about undefined variable
ENV PYTHONPATH="/app${PYTHONPATH:+:$PYTHONPATH}"

# Default command (can be overridden)
CMD ["python", "-m", "mosaic.mosaic"]

