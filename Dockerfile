FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install Python and UV
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

WORKDIR /app

# Copy project files
COPY pyproject.toml uv.lock ./
COPY llm_summarize/ ./llm_summarize/
COPY configs/ ./configs/
COPY scripts/ ./scripts/

# Install dependencies
RUN uv sync

# Set entry point
CMD ["uv", "run", "python", "scripts/run_pipeline.py"]
