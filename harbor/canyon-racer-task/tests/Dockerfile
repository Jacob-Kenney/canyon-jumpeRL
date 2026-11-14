FROM python:3.14-slim
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 && rm -rf /var/lib/apt/lists/*
COPY game /opt/game
RUN pip install --no-cache-dir -c /opt/game/constraints.txt gymnasium mcp pillow pyboy
ENV PYTHONPATH=/opt/game/src CANYON_RACER_ROM=/opt/game/roms/canyon_racer_0.7.7_mbc5_rumble.gb \
    SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy
COPY test.sh replay.py /tests/
RUN chmod +x /tests/test.sh
