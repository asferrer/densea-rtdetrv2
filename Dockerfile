FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends git libgl1 libglib2.0-0 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app/RT-DETR

# torch 2.7.0 and torchvision 0.22.0 (CUDA 12.8) come with the base image.
COPY rtdetrv2_pytorch/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# The code is mounted at /app/RT-DETR by docker-compose.yml.
CMD ["sleep", "infinity"]
