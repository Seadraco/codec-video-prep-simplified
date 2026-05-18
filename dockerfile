ARG PYTHON_IMAGE=python:3.10-bookworm
FROM ${PYTHON_IMAGE}

ARG USE_CN_MIRROR=1
ARG APT_MIRROR=http://mirrors.aliyun.com/debian/
ARG APT_SECURITY_MIRROR=http://mirrors.aliyun.com/debian-security/
ARG PIP_INDEX=https://mirrors.aliyun.com/pypi/simple
ARG PIP_TRUSTED_HOST=mirrors.aliyun.com

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN if [ "$USE_CN_MIRROR" = "1" ]; then \
        sed -i \
          -e "s|http://deb.debian.org/debian-security|${APT_SECURITY_MIRROR}|g" \
          -e "s|http://security.debian.org/debian-security|${APT_SECURITY_MIRROR}|g" \
          -e "s|http://deb.debian.org/debian|${APT_MIRROR}|g" \
          /etc/apt/sources.list.d/debian.sources; \
    fi

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        bzip2 \
        ca-certificates \
        ffmpeg \
        nasm \
        pkg-config \
        yasm \
    && rm -rf /var/lib/apt/lists/*

RUN if [ "$USE_CN_MIRROR" = "1" ]; then \
        pip config set global.index-url "$PIP_INDEX" && \
        pip config set global.trusted-host "$PIP_TRUSTED_HOST" && \
        pip config set global.timeout 300 && \
        pip config set global.retries 5; \
    fi

RUN python -m pip install --upgrade pip setuptools wheel

WORKDIR /workspace

CMD ["bash"]
