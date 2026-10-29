# Worker image for GKE L4 nodes. The TensorRT base image pins CUDA + TensorRT;
# engines must be built with the same TensorRT version (engine-build-job.yaml
# uses this image too).
FROM nvcr.io/nvidia/tensorrt:24.12-py3

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg libgl1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY proto ./proto
COPY scripts ./scripts
COPY src ./src
RUN pip install --upgrade pip \
 && pip install "grpcio-tools>=1.64" \
 && ./scripts/compile_proto.sh \
 && pip install ".[gpu,cloud]"

ARG GIT_COMMIT=unknown
ENV GIT_COMMIT=${GIT_COMMIT}

RUN useradd --uid 10001 --create-home pap && mkdir -p /scratch /models/engines && chown -R pap /scratch /models
USER pap
ENTRYPOINT ["python", "-m", "physical_ai_pipeline.cloud.worker"]
CMD ["--config", "/etc/pap/config.yaml"]
