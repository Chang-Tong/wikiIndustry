FROM python:3.11-slim

WORKDIR /app

ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY
ENV HTTP_PROXY=${HTTP_PROXY}
ENV HTTPS_PROXY=${HTTPS_PROXY}
ENV NO_PROXY=${NO_PROXY}

COPY pyproject.toml /app/pyproject.toml
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG PIP_TRUSTED_HOST=pypi.org
ENV PIP_INDEX_URL=${PIP_INDEX_URL}
ENV PIP_TRUSTED_HOST=${PIP_TRUSTED_HOST}
ENV PIP_DEFAULT_TIMEOUT=120
RUN set -eux; \
  for idx in \
    "${PIP_INDEX_URL}" \
    "https://pypi.tuna.tsinghua.edu.cn/simple" \
    "https://mirrors.aliyun.com/pypi/simple"; do \
    pip install --no-cache-dir --retries 8 -i "${idx}" -e /app \
      && exit 0; \
    echo "pip install failed with index: ${idx}, trying next..."; \
  done; \
  exit 1

COPY app /app/app
COPY agent_mcp /app/agent_mcp

ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "agent_mcp.server"]
