# rom/

Game Boy side of the project. Everything here derives from [Canyon Racer](https://github.com/bbbbbr/canyon-racer) by bbbbbr
(CC-BY-NC-SA 4.0, see the `LICENSE` file in each subfolder).

| Path | What it is |
|---|---|
| `canyon-racer-link/` | Fork of Canyon Racer (v0.7.7, GBDK-2020) that takes button commands from the link port and sends a state packet back. The fork is built with `make LINK=1` and its additions are compiled only when `LINK_INPUT` is defined. Without it the ROM matches the stock build. New file: `src/link.c`/`link.h`. Changed: `Makefile`, `src/gameplay.c`, `src/input.h`, `src/main.c` |
| `tools/` | Host-side Python: `link_client.py` (link protocol framing and transports), `pyboy_link_peer.py` (PyBoy fixture that plays the other end of the cable), `gbdis.py` (tiny disassembler), `tests/` |

`rom/canyon-racer` is a plain clone of the upstream repo, used to build the stock ROM. It is not committed. Re-create it with
`git clone https://github.com/bbbbbr/canyon-racer rom/canyon-racer`.

Building needs GBDK-2020 and make. See the upstream README in `canyon-racer-link/README.md`. The ROM the environment uses is already built
in `../roms/`.
