// Link-cable command/state channel for Canyon Racer
//
// [canyon-jumpeRL change] NEW FILE, not part of upstream Canyon Racer.
//
// Derivative work of Canyon Racer, Copyright 2023 (c) bbbbbr
// (https://github.com/bbbbbr/canyon-racer). Changes (c) 2026 canyon-jumpeRL.
//
// This software is licensed under:
//
// CC-BY-NC-SA: Attribution Non-Commercial Share Alike Creative Commons
// Attribution-NonCommercial-ShareAlike 4.0 International License
// See: http://creativecommons.org/licenses/by-nc-sa/4.0/
//
// For the purposes of this project "Share Alike" shall also include
// distribution of the source code and any changes to it.
//
// Protocol specification: docs/LINK_PROTOCOL.md (canyon-jumpeRL repo).
// Everything here is compiled only when LINK_INPUT is defined
// (make LINK=1 ...). With LINK_INPUT undefined the ROM is byte-identical
// to the stock build.

#ifndef LINK_H
#define LINK_H

#ifdef LINK_INPUT

#include <stdint.h>

// ---- Wire constants (keep in sync with docs/LINK_PROTOCOL.md and rom/tools/link_client.py)
#define LINK_PROTO_VERSION      1u

#define LINK_STATE_SYNC         0xA5u   // first byte of a state packet (GB -> host)
#define LINK_STATE_LEN          30      // total state packet length incl. sync + 2 checksum bytes

#define LINK_CMD_SYNC           0x5Au   // first byte of a command packet (host -> GB)
#define LINK_CMD_LEN            7u      // sync, seq, buttons, hold, flags, chk_a, chk_b

#define LINK_IDLE_BYTE          0x00    // filler the GB sends when it has nothing (ext-clock mode only)

#define LINK_TIMEOUT_TICKS      30u     // ~0.5 s without a valid command -> link inactive, buttons released

// Command flags (byte 4 of command packet)
#define LINK_CMDF_HOLD_POLLS    0x01u   // 'hold' counts game input polls instead of frames (ticks)
#define LINK_CMDF_MASK_JOYPAD   0x02u   // ignore the physical joypad while the link is active

// State flags (byte 4 of state packet)
#define LINK_SF_PAUSED          0x01u
#define LINK_SF_PLAYER_ACTIVE   0x02u
#define LINK_SF_CRASHING        0x04u
#define LINK_SF_CRASH_ENDED     0x08u
#define LINK_SF_GAME_OVER       0x10u
#define LINK_SF_INPUT_POLLED    0x20u   // game polled input at least once since previous state packet
#define LINK_SF_LINK_ACTIVE     0x40u   // a valid command was received within LINK_TIMEOUT_TICKS
#define LINK_SF_RESTORING       0x80u   // rewind/restore sequence in progress

// Serial control values (no 'u' suffix on constants used inside __asm blocks)
#ifdef LINK_EXT_CLOCK
    #define LINK_SC_ARM         0x80    // transfer enable, EXTERNAL clock (peer is clock master)
#else
    #define LINK_SC_ARM         0x81    // transfer enable, INTERNAL clock 8192 Hz (GB is clock master)
#endif

#define LINK_RX_RING_SZ         64      // must be a power of 2
#define LINK_RX_MASK            63      // LINK_RX_RING_SZ - 1

void link_init(void);
void link_frame(void);
uint8_t link_merge_keys(uint8_t joy);

extern volatile uint8_t link_restoring;

#endif // LINK_INPUT

#endif // LINK_H
