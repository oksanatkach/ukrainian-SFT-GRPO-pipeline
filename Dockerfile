FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
FROM ghcr.io/pytorch/pytorch:2.8.0-cuda12.8-cudnn8-runtime

# Define build-time arguments (placeholders)
ARG HF_API_TOKEN
ARG OPENAI_API_KEY

# Expose them as environment variables in the container
ENV HF_API_TOKEN=${HF_API_TOKEN}
ENV OPENAI_API_KEY=${OPENAI_API_KEY}

# Install Python and UV
RUN apt-get update && apt-get install -y \
    python3.13 \
    python3-pip \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
RUN source $HOME/.local/bin/env

WORKDIR /app

RUN git clone https://github.com/oksanatkach/ukrainian-SFT-GRPO-pipeline.git
RUN cd ukrainian-SFT-GRPO-pipeline

# Install dependencies
RUN uv sync

# Set entry point
CMD ["uv", "run", "python", "scripts/run_pipeline.py"]
