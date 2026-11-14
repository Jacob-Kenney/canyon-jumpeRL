# Agent container: NO game code, NO ROM. Only an MCP client for scripting; the game lives in the mcp-server sidecar.
FROM python:3.14-slim
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates git bash && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir mcp==2.3.0 numpy==2.5.4
WORKDIR /app
