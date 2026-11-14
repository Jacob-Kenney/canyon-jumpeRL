"""Host-side library for the Canyon Racer link-cable protocol (docs/LINK_PROTOCOL.md).

The Game Boy ROM built with ``make link`` streams 30-byte *state packets* out of its
serial port and accepts 7-byte *command packets* in. This module implements the
framing on top of an abstract byte transport, so the same code runs against:

* the PyBoy verification fixture (``pyboy_link_peer.EmuTransport``),
* a USB-serial bridge / microcontroller wired to the Game Boy link port
  (``SerialBridgeTransport``),
* a TCP (or WebSocket) relay that carries the raw link byte stream
  (``TCPTransport``; WebSocket notes below).

Wire summary (all multi-byte values little-endian, checksums Fletcher-style mod 256)::

    state  (GB -> host): A5 tick_lo tick_hi mode flags ship_state ship_counter
                         x_hi x_lo y_hi y_lo z_hi z_lo lives_bcd score0 score1 score2
                         level speed canyon_x obs_count o0_y o0_x o0_type o1_y o1_x o1_type
                         ack_seq chk_a chk_b                                   (30 bytes)
    command(host -> GB): 5A seq buttons hold flags chk_a chk_b                 (7 bytes)

Transport contract
------------------
A link cable is full duplex and *clocked*: every time a byte is clocked, one byte moves
in each direction. A transport therefore exposes two queues:

``write(data)``
    Queue bytes to be shifted INTO the Game Boy, in order, one per clocked exchange.
    When the queue is empty the adapter must answer with ``0x00`` (idle). It must NOT
    answer ``0xFF``: that is what the Game Boy reads from an unplugged cable.
``read(max_bytes, timeout)``
    Return the bytes that were shifted OUT of the Game Boy, in order (possibly empty).
``clock(nbytes)``
    Only for adapters that are the clock master (ROM built with ``LINK_EXT_CLOCK``):
    request ``nbytes`` exchanges. In the default build the Game Boy is the clock master
    and this is a no-op.

Plugging in real hardware
-------------------------
(i) USB-serial link adapter / microcontroller (hardware-independent of Shiny Relay).
    Wire the GB link port (SCK, SIN, SOUT, GND; 5 V logic on DMG/GBC-class hardware,
    check the Chromatic's levels before connecting a 3.3 V MCU) to an MCU SPI peripheral
    in SPI *slave* mode 3 (clock idles high, data changes on falling edge, sampled on
    rising edge, MSB first) for the default ROM build. Firmware loop: on every completed
    SPI byte, push the received byte to the USB-serial port and load the next byte from
    a FIFO filled by the USB-serial port (``0x00`` when empty) into the SPI data
    register. The GB clocks at 8192 Hz (122 us per bit) and leaves >= ~90 us between
    bytes, which is ample time for an SPI-slave interrupt. ``SerialBridgeTransport``
    then simply is the USB CDC port. For the ``LINK_EXT_CLOCK`` ROM the MCU is SPI
    *master*: clock one byte, then wait >= 1 ms (and ideally >= 2 ms) before the next,
    because the GB re-arms from an interrupt that can be delayed by the VBlank/audio
    handler. Implement ``clock()`` as a "clock N bytes" request to the firmware.
(ii) TCP / WebSocket relay. ``TCPTransport`` carries the raw link byte stream in both
    directions (one TCP byte == one link exchange). A WebSocket relay works the same way
    with binary frames; wrap any object with ``send(bytes)`` / ``recv() -> bytes`` in
    ``CallableTransport``. Whatever sits at the far end must still behave like a
    shift-register peer at the cable: pre-load the next byte before the GB clocks.

Shiny Relay's wire protocol is not publicly documented; nothing here assumes it. See
docs/LINK_PROTOCOL.md "On-device test checklist" before relying on it.
"""

from __future__ import annotations

import socket
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional

# ---------------------------------------------------------------------------------------
# Protocol constants (must match rom/canyon-racer-link/src/link.h)
# ---------------------------------------------------------------------------------------
PROTO_VERSION = 1

STATE_SYNC = 0xA5
STATE_LEN = 30
CMD_SYNC = 0x5A
CMD_LEN = 7
IDLE_BYTE = 0x00
NO_PEER_BYTE = 0xFF          # what a Game Boy reads from an unplugged cable
LINK_TIMEOUT_TICKS = 30      # ROM releases link buttons after this many ticks w/o a valid command

# GBDK joypad bits
RIGHT, LEFT, UP, DOWN = 0x01, 0x02, 0x04, 0x08
A, B, SELECT, START = 0x10, 0x20, 0x40, 0x80
BUTTONS = {"right": RIGHT, "left": LEFT, "up": UP, "down": DOWN,
           "a": A, "b": B, "select": SELECT, "start": START}

# Command flags
CMDF_HOLD_POLLS = 0x01       # 'hold' counts game input polls instead of ticks
CMDF_MASK_JOYPAD = 0x02      # ignore physical joypad while the link is active

# State flags
SF_PAUSED = 0x01
SF_PLAYER_ACTIVE = 0x02
SF_CRASHING = 0x04
SF_CRASH_ENDED = 0x08
SF_GAME_OVER = 0x10
SF_INPUT_POLLED = 0x20
SF_LINK_ACTIVE = 0x40
SF_RESTORING = 0x80

# Screen / mode byte (game_state in main.c)
MODE_CREDITS, MODE_TITLE, MODE_HELP, MODE_STARTING, MODE_RUNNING, MODE_GAME_OVER = range(6)
MODE_NAMES = {0: "CREDITS", 1: "TITLE", 2: "HELP", 3: "STARTING", 4: "RUNNING", 5: "GAME_OVER"}

# ship_state values
SHIP_ON_GROUND, SHIP_JUMP, SHIP_CRASHING, SHIP_CRASH_ENDED = 0x10, 0x11, 0x22, 0x23


def fletcher8(data: Iterable[int]) -> tuple[int, int]:
    """Fletcher-style checksum used by both packet types: a = sum(b), s = sum(a), mod 256."""
    a = s = 0
    for b in data:
        a = (a + b) & 0xFF
        s = (s + a) & 0xFF
    return a, s


def bcd_to_int(raw: bytes) -> int:
    """Little-endian packed BCD -> int."""
    value = 0
    for b in reversed(raw):
        value = value * 100 + (b >> 4) * 10 + (b & 0x0F)
    return value


# ---------------------------------------------------------------------------------------
# Packets
# ---------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Obstacle:
    y: int          # top edge, screen pixels
    x: int          # left edge of the 32 px wide obstacle sprite, screen pixels
    type: int       # raw type: bits0-2 0=LEFT 1=RIGHT 2=MIDDLE 3=FULL 4=ITEM(+1 rewind); bit7 = hidden


@dataclass(frozen=True)
class StatePacket:
    raw: bytes
    tick: int
    mode: int
    flags: int
    ship_state: int
    ship_counter: int
    ship_x_raw: int     # 8.8 fixed point, screen pixels
    ship_y_raw: int
    ship_z_raw: int
    lives_bcd: int
    score_bcd: bytes    # 3 bytes, little endian packed BCD
    level: int
    speed_raw: int      # obstacle speed >> 2  (units of 1/64 px per frame)
    canyon_x: int       # canyon left edge at ship centre row
    obstacle_count: int
    obstacles: tuple
    ack_seq: int

    @property
    def mode_name(self) -> str:
        return MODE_NAMES.get(self.mode, f"?{self.mode}")

    @property
    def ship_x(self) -> float:
        return self.ship_x_raw / 256.0

    @property
    def ship_y(self) -> float:
        return self.ship_y_raw / 256.0

    @property
    def ship_z(self) -> float:
        return self.ship_z_raw / 256.0

    @property
    def lives(self) -> int:
        return (self.lives_bcd >> 4) * 10 + (self.lives_bcd & 0x0F)

    @property
    def score(self) -> int:
        return bcd_to_int(self.score_bcd)

    def has(self, flag: int) -> bool:
        return bool(self.flags & flag)

    @property
    def paused(self) -> bool:
        return self.has(SF_PAUSED)

    @property
    def game_over(self) -> bool:
        return self.has(SF_GAME_OVER)

    @property
    def input_polled(self) -> bool:
        return self.has(SF_INPUT_POLLED)

    @property
    def link_active(self) -> bool:
        return self.has(SF_LINK_ACTIVE)

    @classmethod
    def decode(cls, raw: bytes) -> "StatePacket":
        if len(raw) != STATE_LEN or raw[0] != STATE_SYNC:
            raise ValueError("not a state packet")
        if fletcher8(raw[:-2]) != (raw[-2], raw[-1]):
            raise ValueError("bad checksum")
        obstacles = []
        for i in range(2):
            y, x, t = raw[21 + 3 * i: 24 + 3 * i]
            if i < raw[20]:
                obstacles.append(Obstacle(y, x, t))
        return cls(
            raw=bytes(raw),
            tick=raw[1] | raw[2] << 8,
            mode=raw[3],
            flags=raw[4],
            ship_state=raw[5],
            ship_counter=raw[6],
            ship_x_raw=raw[7] << 8 | raw[8],
            ship_y_raw=raw[9] << 8 | raw[10],
            ship_z_raw=raw[11] << 8 | raw[12],
            lives_bcd=raw[13],
            score_bcd=bytes(raw[14:17]),
            level=raw[17],
            speed_raw=raw[18],
            canyon_x=raw[19],
            obstacle_count=raw[20],
            obstacles=tuple(obstacles),
            ack_seq=raw[27],
        )


def encode_command(seq: int, buttons: int, hold: int = 0, flags: int = 0) -> bytes:
    body = bytes([CMD_SYNC, seq & 0xFF, buttons & 0xFF, hold & 0xFF, flags & 0xFF])
    return body + bytes(fletcher8(body))


def encode_state(fields: bytes) -> bytes:
    """Build a state packet from bytes 1..27 (used by tests / simulators)."""
    if len(fields) != STATE_LEN - 3:
        raise ValueError("need 27 field bytes")
    body = bytes([STATE_SYNC]) + bytes(fields)
    return body + bytes(fletcher8(body))


class StateDecoder:
    """Incremental decoder for the GB -> host byte stream.

    Hunts for the sync byte, validates length + checksum, and on any failure resumes the
    search at the byte after the false sync, so arbitrary garbage (0xFF from a dead line,
    idle 0x00, dropped or corrupted bytes) is skipped without losing a following good
    packet.
    """

    def __init__(self) -> None:
        self.buf = bytearray()
        self.packets_ok = 0
        self.checksum_errors = 0
        self.bytes_skipped = 0

    def feed(self, data: bytes) -> List[StatePacket]:
        self.buf.extend(data)
        out: List[StatePacket] = []
        while True:
            i = self.buf.find(STATE_SYNC)
            if i < 0:
                self.bytes_skipped += len(self.buf)
                self.buf.clear()
                break
            if i:
                self.bytes_skipped += i
                del self.buf[:i]
            if len(self.buf) < STATE_LEN:
                break
            cand = bytes(self.buf[:STATE_LEN])
            try:
                pkt = StatePacket.decode(cand)
            except ValueError:
                self.checksum_errors += 1
                self.bytes_skipped += 1
                del self.buf[:1]
                continue
            del self.buf[:STATE_LEN]
            self.packets_ok += 1
            out.append(pkt)
        return out


# ---------------------------------------------------------------------------------------
# Transports
# ---------------------------------------------------------------------------------------
class Transport:
    """Abstract full-duplex clocked byte transport (see module docstring)."""

    def write(self, data: bytes) -> None:
        raise NotImplementedError

    def read(self, max_bytes: int = 4096, timeout: float = 0.0) -> bytes:
        raise NotImplementedError

    def clock(self, nbytes: int) -> None:  # only for host-clocked (LINK_EXT_CLOCK) setups
        return None

    def close(self) -> None:
        return None


class SerialBridgeTransport(Transport):
    """USB-serial bridge (MCU acting as the GB's link-port peer). Requires ``pyserial``.

    The bridge firmware must forward raw link bytes both ways (see module docstring).
    If the firmware is the clock master (ext-clock ROM), ``clock_cmd`` builds the request
    it understands, e.g. ``lambda n: b"C" + bytes([n])``.
    """

    def __init__(self, port: str, baudrate: int = 115200,
                 clock_cmd: Optional[Callable[[int], bytes]] = None) -> None:
        import serial  # type: ignore  # pyserial, optional dependency

        self._ser = serial.Serial(port, baudrate=baudrate, timeout=0)
        self._clock_cmd = clock_cmd

    def write(self, data: bytes) -> None:
        self._ser.write(data)

    def read(self, max_bytes: int = 4096, timeout: float = 0.0) -> bytes:
        self._ser.timeout = timeout
        return self._ser.read(max_bytes)

    def clock(self, nbytes: int) -> None:
        if self._clock_cmd is not None:
            self._ser.write(self._clock_cmd(nbytes))

    def close(self) -> None:
        self._ser.close()


class TCPTransport(Transport):
    """Raw link byte stream over TCP (one byte per link exchange in each direction)."""

    def __init__(self, host: str, port: int, connect_timeout: float = 5.0) -> None:
        self._sock = socket.create_connection((host, port), timeout=connect_timeout)
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def write(self, data: bytes) -> None:
        self._sock.sendall(data)

    def read(self, max_bytes: int = 4096, timeout: float = 0.0) -> bytes:
        self._sock.settimeout(timeout if timeout > 0 else 0.0)
        try:
            return self._sock.recv(max_bytes)
        except (BlockingIOError, socket.timeout):
            return b""

    def close(self) -> None:
        self._sock.close()


class CallableTransport(Transport):
    """Adapter for e.g. a WebSocket client: ``send(bytes)`` and ``recv(timeout) -> bytes``."""

    def __init__(self, send: Callable[[bytes], None], recv: Callable[[float], bytes],
                 clock: Optional[Callable[[int], None]] = None) -> None:
        self._send, self._recv, self._clock = send, recv, clock

    def write(self, data: bytes) -> None:
        self._send(data)

    def read(self, max_bytes: int = 4096, timeout: float = 0.0) -> bytes:
        return self._recv(timeout)[:max_bytes]

    def clock(self, nbytes: int) -> None:
        if self._clock:
            self._clock(nbytes)


# ---------------------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------------------
@dataclass
class LinkClient:
    transport: Transport
    decoder: StateDecoder = field(default_factory=StateDecoder)
    seq: int = 0
    last_cmd: Optional[bytes] = None
    latest: Optional[StatePacket] = None
    history: List[StatePacket] = field(default_factory=list)
    keep_history: bool = False

    # ---- commands -------------------------------------------------------------------
    def send(self, buttons: int, hold: int = 0, flags: int = 0) -> int:
        """Send a NEW command (fresh seq). Returns the seq used.

        hold = 0     -> buttons stay asserted until the next command or the link timeout
        hold = n > 0 -> buttons asserted for n ticks (or n game input polls with
                        CMDF_HOLD_POLLS), then released by the ROM itself.
        """
        self.seq = (self.seq + 1) & 0xFF
        self.last_cmd = encode_command(self.seq, buttons, hold, flags)
        self.transport.write(self.last_cmd)
        return self.seq

    def keepalive(self) -> None:
        """Resend the last command unchanged. The ROM treats a repeated seq as a
        keep-alive only (idempotent), so taps are never applied twice."""
        if self.last_cmd is None:
            self.send(0)
        else:
            self.transport.write(self.last_cmd)

    def hold(self, buttons: int, flags: int = 0) -> int:
        return self.send(buttons, 0, flags)

    def release(self, flags: int = 0) -> int:
        return self.send(0, 0, flags)

    def tap(self, buttons: int, polls: int = 1, flags: int = 0) -> int:
        """Press for exactly `polls` game input polls (guaranteed to be seen by a menu
        that is polling), then release."""
        return self.send(buttons, polls, flags | CMDF_HOLD_POLLS)

    def press_frames(self, buttons: int, frames: int, flags: int = 0) -> int:
        return self.send(buttons, frames, flags)

    # ---- state ----------------------------------------------------------------------
    def poll(self, timeout: float = 0.0) -> List[StatePacket]:
        data = self.transport.read(4096, timeout)
        pkts = self.decoder.feed(data) if data else []
        if pkts:
            self.latest = pkts[-1]
            if self.keep_history:
                self.history.extend(pkts)
        return pkts

    def acked(self, seq: Optional[int] = None) -> bool:
        seq = self.seq if seq is None else seq
        return self.latest is not None and self.latest.ack_seq == seq

    def wait_for(self, predicate: Callable[[StatePacket], bool], timeout: float = 5.0,
                 step: Optional[Callable[[], None]] = None) -> Optional[StatePacket]:
        """Poll until a packet satisfies predicate. `step` advances time for simulated
        transports (e.g. ticks the emulator); real transports just wait on read()."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if step:
                step()
            for p in self.poll(0.0 if step else 0.02):
                if predicate(p):
                    return p
        return None
