FROM python:3.12-slim

ARG TA_LIB_VERSION=0.6.4

WORKDIR /app

# Install TA-Lib C library early so this layer can be reused across code-only changes.
# Installation follows upstream Debian package guidance from ta-lib.org/install.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    p7zip-full \
    postgresql-client \
    wget \
    && ARCH="$(dpkg --print-architecture)" \
    && case "${ARCH}" in \
        amd64|arm64|i386) TA_ARCH="${ARCH}" ;; \
        *) echo "Unsupported TA-Lib architecture: ${ARCH}" >&2; exit 1 ;; \
    esac \
    && wget -O /tmp/ta-lib.deb \
    "https://github.com/TA-Lib/ta-lib/releases/download/v${TA_LIB_VERSION}/ta-lib_${TA_LIB_VERSION}_${TA_ARCH}.deb" \
    && apt-get install -y --no-install-recommends /tmp/ta-lib.deb \
    && rm -f /tmp/ta-lib.deb \
    && apt-get purge -y --auto-remove wget \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Install Python dependencies before copying app source for better layer caching.
# `uv.lock` may be absent in local branches; resolve from pyproject when needed.
COPY pyproject.toml ./
RUN uv sync --no-dev --no-install-project

# Copy project files
COPY apps ./apps
COPY packages ./packages

# Expose port
EXPOSE 8000

# Run the application
CMD ["uv", "run", "uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
