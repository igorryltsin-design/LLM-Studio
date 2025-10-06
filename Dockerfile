# syntax=docker/dockerfile:1.6

ARG NODE_VERSION=20

FROM node:${NODE_VERSION}-bullseye AS frontend-builder
WORKDIR /app
ENV CI=1

COPY package.json package-lock.json ./
RUN npm ci

COPY . .
RUN npm run build

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10 \
    HF_HOME=/opt/llm-studio/cache/huggingface \
    TRANSFORMERS_CACHE=/opt/llm-studio/cache/huggingface \
    TORCH_HOME=/opt/llm-studio/cache/torch

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        python3-dev \
        build-essential \
        git \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/llm-studio

COPY . .

RUN python3 -m pip install --upgrade pip \
    && if [ -d vendor/pip ] && find vendor/pip -maxdepth 1 -name 'torch-*.whl' | grep -q .; then \
         if ! python3 -m pip install --no-index --find-links=vendor/pip "torch==2.4.0"; then \
           python3 -m pip install --no-cache-dir "torch==2.4.0" --index-url https://download.pytorch.org/whl/cu121; \
         fi; \
       else \
         python3 -m pip install --no-cache-dir "torch==2.4.0" --index-url https://download.pytorch.org/whl/cu121; \
       fi \
    && if [ -d vendor/pip ] && [ "$(ls -A vendor/pip 2>/dev/null)" ]; then \
         if ! python3 -m pip install --no-index --find-links=vendor/pip -r requirements.txt; then \
           python3 -m pip install --no-cache-dir -r requirements.txt; \
         fi; \
       else \
         python3 -m pip install --no-cache-dir -r requirements.txt; \
       fi \
    && rm -rf vendor/pip

RUN rm -rf dist \
    && mkdir -p cache/huggingface cache/torch Models \
    && chown -R root:root /opt/llm-studio

COPY --from=frontend-builder /app/dist ./dist

RUN groupadd -r llm && useradd -r -g llm -d /opt/llm-studio llm \
    && chown -R llm:llm /opt/llm-studio

VOLUME ["/opt/llm-studio/Models"]

EXPOSE 8001

ENV UVICORN_APP=base_model_server:app \
    UVICORN_HOST=0.0.0.0 \
    UVICORN_PORT=8001 \
    UVICORN_WORKERS=1

COPY docker-entrypoint.sh ./docker-entrypoint.sh
RUN chmod +x ./docker-entrypoint.sh

USER llm

ENTRYPOINT ["./docker-entrypoint.sh"]
CMD []
