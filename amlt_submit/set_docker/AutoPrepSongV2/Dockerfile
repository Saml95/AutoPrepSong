FROM nvcr.io/nvidia/pytorch:25.11-py3

# Install system dependencies
RUN apt-get update && apt-get install -y \
    openmpi-bin \
    libopenmpi-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Uninstall incompatible flash-attn from base image
RUN pip uninstall flash-attn -y || true

# Copy and install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt