FROM debian:trixie-slim
ARG INSTALL_LITERT=1
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 \
    HAILO_CONFIG=/etc/hailo-10h-services.yaml HOME=/var/lib/hailo-10h-services
# Vendor runtime only: the matching kernel driver belongs on the Docker host.
COPY deploy/vendor/ /tmp/vendor/
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-venv python3-pip libsndfile1 ca-certificates libgomp1 \
    && test "$(find /tmp/vendor -name '*.deb' | wc -l)" -eq 1 \
    && test "$(find /tmp/vendor -name '*.whl' | wc -l)" -eq 1 \
    && apt-get install -y /tmp/vendor/*.deb \
    && ldconfig \
    && python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install /tmp/vendor/*.whl \
    && rm -rf /tmp/vendor /var/lib/apt/lists/*
COPY pyproject.toml README.md /opt/source/
COPY src /opt/source/src
RUN /opt/venv/bin/pip install /opt/source \
    && if [ "$INSTALL_LITERT" = 1 ]; then /opt/venv/bin/pip install litert-lm; fi \
    && /opt/venv/bin/python -c 'from hailo_platform.genai import VLM, Speech2Text' \
    && mkdir -p /var/lib/hailo-10h-services /usr/local/hailo/resources/models/hailo10h
WORKDIR /var/lib/hailo-10h-services
EXPOSE 8090 10300
ENTRYPOINT ["/opt/venv/bin/hailo-10h-services"]
