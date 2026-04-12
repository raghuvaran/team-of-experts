FROM ghcr.io/open-webui/open-webui:main

# Install TOXP (context is project root, set by docker-compose)
COPY pyproject.toml README.md /tmp/toxp-src/
COPY toxp/ /tmp/toxp-src/toxp/
RUN pip install --no-cache-dir /tmp/toxp-src && rm -rf /tmp/toxp-src

# Embed the pipe function so start.sh can register it
COPY ui/toxp_pipe.py /app/toxp_pipe.py
