"""PyBoy verification fixture: emulates the OTHER end of the Game Boy link cable.

What it does
------------
PyBoy 2.9.0 ships a link-cable model (``pyboy.core.serial.SerialSharedMemory``) that is
normally used to connect two PyBoy instances through a 2-byte shared-memory block and a
barrier. In bit mode (``serial_interrupt_based=False``) the emulated SB/SC hardware does,
for every serial clock of a transfer::

    write(my_slot, SB.bit7)   # GB drives SOUT
    synchronize()             # clock edge, both sides have presented their bit
    SB = SB << 1 | read(other_slot)   # GB samples SIN
    synchronize()

and after 8 clocks clears SC.7 and raises the serial interrupt (IF.3). The "shared
memory" is just a duck-typed Python object, so this module supplies one that is a
cable-level peer: an 8-bit shift register that

* presents the MSB of its pre-loaded byte on the line at each clock,
* shifts in the GB's SOUT bit,
* loads its next byte from a FIFO only at a byte boundary (i.e. the byte the GB receives
  is the one the peer had loaded BEFORE the GB started clocking it),
* answers 0x00 when its FIFO is empty (idle), and
* when "unplugged", drives the line high (GB reads 0xFF) and hears nothing.

The ROM under test is the exact binary built for the device; nothing in the game is
poked. All exchanges go through the emulator's own SB/SC/IF serial logic.

What it does NOT prove (see docs/LINK_PROTOCOL.md "Verification limits")
-------------------------------------------------------------------------
* PyBoy 2.9.0 uses 128 T-cycles per serial bit (CYCLES_8192HZ = 128) although the real
  internal clock is 8192 Hz = 512 T-cycles per bit. Transfers in the emulator are 4x
  faster than on hardware; byte throughput / packet rate numbers from the emulator are
  therefore optimistic by up to 4x and the hardware budget is computed analytically.
* The first-bit phase alignment to the DIV-derived serial clock is not modelled.
* For the LINK_EXT_CLOCK build, PyBoy's model starts clocking 128 T-cycles after the GB
  arms SC=0x80 (as if the peer were always waiting), so a peer that clocks *before* the
  GB has re-armed cannot be reproduced here.
* Electrical behaviour (levels, cable capacitance, Chromatic link port, Shiny Relay).
"""

from __future__ import annotations

import collections
import os
import re
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Optional

from pyboy import PyBoy

from link_client import Transport

HERE = os.path.dirname(os.path.abspath(__file__))
ROM_DIR = os.path.normpath(os.path.join(HERE, "..", "canyon-racer-link", "build", "gb"))
STOCK_DIR = os.path.normpath(os.path.join(HERE, "..", "canyon-racer", "build", "gb"))

PYBOY_T_CYCLES_PER_BIT = 128       # PyBoy 2.9.0 value
HW_T_CYCLES_PER_BIT = 512          # 4194304 Hz / 8192 Hz


def rom_path(name: str) -> str:
    return os.path.join(ROM_DIR, name)


LINK_ROM = rom_path("canyon_racer_0.7.7_mbc5_link.gb")
LINK_EXTCLK_ROM = rom_path("canyon_racer_0.7.7_mbc5_link_extclk.gb")
FLAGOFF_MBC5_ROM = rom_path("canyon_racer_0.7.7_mbc5.gb")
FLAGOFF_RUMBLE_ROM = rom_path("canyon_racer_0.7.7_mbc5_rumble.gb")
STOCK_RUMBLE_ROM = os.path.join(STOCK_DIR, "canyon_racer_0.7.7_mbc5_rumble.gb")


def load_symbols(rom: str) -> Dict[str, int]:
    """Parse the GBDK .noi file next to a ROM ("DEF name 0xADDR")."""
    syms: Dict[str, int] = {}
    with open(os.path.splitext(rom)[0] + ".noi") as f:
        for line in f:
            m = re.match(r"DEF (\S+) 0x([0-9A-Fa-f]+)", line)
            if m:
                syms[m.group(1)] = int(m.group(2), 16)
    return syms


# Offsets inside `game_state_data state` (verified against the .cdb debug info of the build)
STATE_OFF = {
    "paused": 2, "game_level": 3, "cur_level.obst_speed": 8,
    "ship_x": 15, "ship_y": 17, "ship_z": 19, "ship_state": 21, "ship_counter": 22,
    "obstacles": 25, "obstacles_active_first": 40, "obstacles_active_count": 42,
    "p_scx_table_frame_base": 65, "score": 77,
}


class ShiftRegisterPeer:
    """Duck-typed replacement for pyboy.core.serial.SerialSharedMemoryBuffer.

    Slot 0 belongs to the emulated Game Boy, slot 1 to this peer.
    """

    def __init__(self) -> None:
        self._init_done = False
        self._phase_a = True
        self._gb_bit = 1
        self._out_bit = 1
        self._bit_index = 0
        self._tx_shift = 0xFF
        self._rx_shift = 0
        self.connected = True              # False = cable unplugged (line pulled high)
        self.tx_fifo: Deque[int] = collections.deque()
        self.rx: Deque[int] = collections.deque()      # bytes received from the GB
        self.rx_log: List[int] = []                     # everything ever received
        self.exchanges = 0
        self.bits = 0
        # Fault injection (host <- GB direction): callable(byte) -> Optional[int]
        # returning None drops the byte, an int replaces it.
        self.rx_filter: Optional[Callable[[int], Optional[int]]] = None
        self.on_byte: Optional[Callable[[int, int], None]] = None

    # --- interface used by SerialSharedMemory -----------------------------------------
    def read(self, slot: int) -> int:
        if not self._init_done:
            return 0  # tells the emulator it is the first party -> takes slot 0
        return self._out_bit

    def write(self, slot: int, value: int) -> None:
        if self._init_done and slot == 0:
            self._gb_bit = value & 1

    def synchronize(self) -> None:
        if not self._init_done:
            self._init_done = True
            return
        if self._phase_a:
            self._clock_edge()
        self._phase_a = not self._phase_a

    # --- shift register ----------------------------------------------------------------
    def _clock_edge(self) -> None:
        self.bits += 1
        if self._bit_index == 0:
            # Byte boundary: the byte that goes out is the one pre-loaded before this clock
            if self.connected:
                self._tx_shift = self.tx_fifo.popleft() if self.tx_fifo else 0x00
            else:
                self._tx_shift = 0xFF
        self._out_bit = (self._tx_shift >> 7) & 1 if self.connected else 1
        self._tx_shift = (self._tx_shift << 1) & 0xFF
        self._rx_shift = ((self._rx_shift << 1) | self._gb_bit) & 0xFF
        self._bit_index += 1
        if self._bit_index == 8:
            self._bit_index = 0
            self.exchanges += 1
            b = self._rx_shift
            if self.on_byte:
                self.on_byte(self.exchanges, b)
            if not self.connected:
                return
            self.rx_log.append(b)
            if self.rx_filter is not None:
                fb = self.rx_filter(b)
                if fb is None:
                    return
                b = fb
            self.rx.append(b)

    # --- helpers -----------------------------------------------------------------------
    def queue(self, data: bytes) -> None:
        self.tx_fifo.extend(data)

    def take_rx(self, max_bytes: int = 1 << 30) -> bytes:
        out = bytearray()
        while self.rx and len(out) < max_bytes:
            out.append(self.rx.popleft())
        return bytes(out)

    def close(self) -> None:  # called by PyBoy on stop
        pass


@dataclass
class Emu:
    """A PyBoy instance optionally connected to a ShiftRegisterPeer."""
    rom: str
    peer: Optional[ShiftRegisterPeer] = None
    cgb: bool = False
    pyboy: PyBoy = field(init=False)
    syms: Dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        kwargs = dict(window="null", sound_emulated=False, cgb=self.cgb, log_level="ERROR")
        if self.peer is not None:
            kwargs.update(serial_shared_memory=self.peer, serial_interrupt_based=False)
        # Fresh, empty cartridge RAM so every run starts from the same save state
        import io
        self.pyboy = PyBoy(self.rom, ram_file=io.BytesIO(b"\x00" * 0x2000), **kwargs)
        self.pyboy.set_emulation_speed(0)
        self.syms = load_symbols(self.rom)

    # --- memory helpers ----------------------------------------------------------------
    def sym(self, name: str) -> int:
        return self.syms[name]

    def u8(self, addr: int) -> int:
        return self.pyboy.memory[addr]

    def u16(self, addr: int) -> int:
        return self.pyboy.memory[addr] | self.pyboy.memory[addr + 1] << 8

    def var(self, name: str) -> int:
        return self.u8(self.sym(name))

    def state_u8(self, field_name: str, extra: int = 0) -> int:
        return self.u8(self.sym("_state") + STATE_OFF[field_name] + extra)

    def state_u16(self, field_name: str) -> int:
        return self.u16(self.sym("_state") + STATE_OFF[field_name])

    def game_snapshot(self) -> dict:
        """Ground-truth game variables from RAM (layout-independent across builds)."""
        st = self.sym("_state")
        m = self.pyboy.memory
        return {
            "game_state": m[self.sym("_game_state")],
            "paused": m[st + 2],
            "level": m[st + 3],
            "ship_x": m[st + 15] | m[st + 16] << 8,
            "ship_y": m[st + 17] | m[st + 18] << 8,
            "ship_z": m[st + 19] | m[st + 20] << 8,
            "ship_state": m[st + 21],
            "ship_counter": m[st + 22],
            "obstacles": tuple(m[st + 25 + i] for i in range(15)),
            "obs_first": m[st + 40], "obs_count": m[st + 42],
            "score": tuple(m[st + 77 + i] for i in range(4)),
            "lives": m[self.sym("_lives_count")],
            "keys": m[self.sym("_keys")],
        }

    def tick(self, frames: int = 1) -> None:
        self.pyboy.tick(frames, False, False)

    def stop(self) -> None:
        self.pyboy.stop(save=False)


class EmuTransport(Transport):
    """link_client Transport backed by the emulator + ShiftRegisterPeer.

    Time only advances when the test calls step()/Emu.tick(); read()/write() never block.
    """

    def __init__(self, emu: Emu) -> None:
        assert emu.peer is not None
        self.emu = emu
        self.peer = emu.peer

    def write(self, data: bytes) -> None:
        self.peer.queue(data)

    def read(self, max_bytes: int = 4096, timeout: float = 0.0) -> bytes:
        return self.peer.take_rx(max_bytes)

    def step(self, frames: int = 1) -> None:
        self.emu.tick(frames)
