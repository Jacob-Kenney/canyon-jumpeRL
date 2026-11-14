# rom/tools

Host-side helpers for the link-cable fork in `../canyon-racer-link`.

- `link_client.py`: framing for the link protocol. State packets are 30 bytes (Game Boy to host) and command packets are 7 bytes (host to Game Boy).
  It runs over an abstract byte transport, with adapters for the PyBoy fixture, a USB-serial bridge, and TCP.
  Constants must match `../canyon-racer-link/src/link.h`.
- `pyboy_link_peer.py`: verification fixture that emulates the other end of the cable using PyBoy's serial model.
- `gbdis.py`: `python gbdis.py ROM START_HEX END_HEX` prints a disassembly range using PyBoy's opcode table.
- `tests/`: tests for the above.

These need the project environment (`uv sync` at the repo root).
