# syntax=docker/dockerfile:1
FROM ubuntu:22.04@sha256:45e9ddc6895604674618a409ca9f73d301da8fdb75fe330e8e9a27b3b219e4ad

# Pass the directory produced by scripts/download_runtime.sh conda as a named context.
RUN --mount=type=bind,from=torch1_snapshot,target=/snapshot \
    cd /snapshot && sha256sum -c torch1-conda-linux-amd64.tar.gz.sha256 && \
    mkdir -p /opt/torch1 && tar -xzf torch1-conda-linux-amd64.tar.gz -C /opt/torch1 && \
    /opt/torch1/bin/python /opt/torch1/bin/conda-unpack

ENV PATH=/opt/torch1/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/tmp/ppred-cache/huggingface \
    MPLCONFIGDIR=/tmp/ppred-cache/matplotlib \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /app
COPY . /app
CMD ["python", "-m", "ppred", "--help"]
