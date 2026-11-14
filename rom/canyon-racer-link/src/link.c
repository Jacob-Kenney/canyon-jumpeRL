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
// Overview (full spec: docs/LINK_PROTOCOL.md)
//
// * Default build (LINK_INPUT): the Game Boy is the serial CLOCK MASTER
//   (SC = 0x81, internal 8192 Hz clock). Once per game tick (= every return
//   from vsync()) link_frame() takes a snapshot of the game state into a
//   30 byte packet and starts clocking it out if the previous packet has
//   finished. Each completed byte raises the serial interrupt; the ISR stores
//   the byte the peer shifted in, pre-loads the next outgoing byte into SB and
//   restarts the clock. The peer (host adapter) is a passive shift register:
//   whatever it pre-loaded before the GB clocks is what the GB receives.
//
// * Optional build (LINK_INPUT + LINK_EXT_CLOCK): the Game Boy is the clock
//   SLAVE (SC = 0x80). The peer generates the clock whenever it wants a byte;
//   the ISR pre-loads the next byte and re-arms. Between packets the GB sends
//   LINK_IDLE_BYTE.
//
// * The serial vector uses ISR_NESTED_VECTOR (EI before the JP), so the
//   timing critical LCD/STAT parallax ISR (and VBlank) can pre-empt the serial
//   ISR. The serial ISR never runs with interrupts disabled for more than the
//   vector dispatch + EI + JP.
//
// * Lost serial interrupts (GBDK set_interrupts() writes IF=0 and the map fx
//   VBlank ISR does a read-modify-write of IF) are detected and replayed by a
//   watchdog in link_frame().

#ifdef LINK_INPUT

#include <gbdk/platform.h>
#include <gb/isr.h>
#include <stdint.h>
#include <stdbool.h>

#include "common.h"
#include "input.h"
#include "lives.h"
#include "entity_ship.h"
#include "link.h"

extern uint8_t game_state; // main.c

// ===== Shared with the serial ISR (names are referenced from asm) =====
uint8_t          link_tx_buf[LINK_STATE_LEN];  // packet being shifted out
volatile uint8_t link_tx_pos;                  // index of next byte to pre-load; >= LEN: packet finished
volatile uint8_t link_busy;                    // a transfer chain is armed (always 1 in ext clock mode)
uint8_t          link_rx_ring[LINK_RX_RING_SZ];
volatile uint8_t link_rx_head;                 // written only by ISR
volatile uint8_t link_isr_count;               // incremented by every ISR run

// ===== Main-loop side state =====
uint8_t  link_rx_tail;
uint16_t link_tick;            // incremented on every link_frame() (one per vsync() return)
uint8_t  link_cmd_buf[LINK_CMD_LEN];
uint8_t  link_cmd_len;
uint8_t  link_last_seq;        // seq of the last applied command
uint8_t  link_keys;            // buttons currently asserted by the link
uint8_t  link_hold;            // remaining ticks/polls for link_keys (0 = until next command)
uint8_t  link_cmd_flags;       // flags of the last applied command
uint8_t  link_joy_mask;        // 0xFF normally, 0x00 when MASK_JOYPAD is active
uint8_t  link_timeout;         // ticks since last valid command (saturating)
uint8_t  link_active;          // a valid command arrived within LINK_TIMEOUT_TICKS
uint8_t  link_polls;           // input polls since the previous state packet
uint8_t  link_rx_errors;       // command packets rejected by checksum (wraps)
uint8_t  link_lost_irq;        // serial completions recovered by the watchdog (wraps)
volatile uint8_t link_restoring; // set by gameplay.c during the rewind sequence


// Serial ISR. Entered from vector 0x58 via ISR_NESTED_VECTOR: "EI ; JP _link_serial_isr",
// so IME is already 1 and higher-priority work (STAT parallax, VBlank) can nest on top.
// Can also be CALLed from main code (watchdog replay); it ends in RETI, which with
// IME=1 in main code is equivalent to RET.
//
// Only AF and HL are used. It must finish all buffer updates BEFORE re-arming SC,
// because once SC is armed the next completion may nest another instance of this ISR
// (only possible in ext clock mode with a very fast peer).
void link_serial_isr(void) __naked {
    __asm
    push af
    push hl

    ; ---- receive: link_rx_ring[link_rx_head] = SB ; link_rx_head = (head + 1) & MASK
    ld   a, (_link_rx_head)
    add  a, #<(_link_rx_ring)
    ld   l, a
    ld   a, #0
    adc  a, #>(_link_rx_ring)
    ld   h, a
    ldh  a, (_SB_REG + 0)
    ld   (hl), a
    ld   a, (_link_rx_head)
    inc  a
    and  a, #LINK_RX_MASK
    ld   (_link_rx_head), a

    ld   hl, #_link_isr_count
    inc  (hl)

    ; ---- transmit: pre-load next byte, then re-arm
    ld   hl, #_link_tx_pos
    ld   a, (hl)
    cp   a, #LINK_STATE_LEN
    jr   nc, 00001$
    inc  (hl)
    add  a, #<(_link_tx_buf)
    ld   l, a
    ld   a, #0
    adc  a, #>(_link_tx_buf)
    ld   h, a
    ld   a, (hl)
    ldh  (_SB_REG + 0), a
    ld   a, #LINK_SC_ARM
    ldh  (_SC_REG + 0), a
    pop  hl
    pop  af
    reti

00001$:
#ifdef LINK_EXT_CLOCK
    ; packet finished: keep answering the clock master with idle bytes
    ld   a, #LINK_IDLE_BYTE
    ldh  (_SB_REG + 0), a
    ld   a, #LINK_SC_ARM
    ldh  (_SC_REG + 0), a
#else
    ; packet finished: stop clocking until the next tick starts a new packet
    xor  a
    ld   (_link_busy), a
#endif
    pop  hl
    pop  af
    reti
    __endasm;
}

// Vector 0x58: EI ; JP _link_serial_isr
ISR_NESTED_VECTOR(VECTOR_SERIAL, link_serial_isr)


// Replacement for vsync(): the Makefile adds -Dvsync=link_vsync for link builds, so
// every vsync() call in the game becomes this. gb.h declares it with
// PRESERVES_REGS(b, c, d, e, h, l), which this wrapper honours.
void link_vsync(void) PRESERVES_REGS(b, c, d, e, h, l) __naked {
    __asm
    push bc
    push de
    push hl
    call _vsync          ; the real GBDK vsync / wait_vbl_done
    call _link_frame
    pop  hl
    pop  de
    pop  bc
    ret
    __endasm;
}


static void link_replay_isr(void) __naked {
    __asm
    jp   _link_serial_isr  ; tail call: its RETI returns to our caller (IME stays 1)
    __endasm;
}


void link_init(void) {
    uint8_t c;
    for (c = 0; c < LINK_STATE_LEN; c++) link_tx_buf[c] = 0;
    link_tx_pos    = LINK_STATE_LEN;  // nothing pending -> first tick builds a packet
    link_rx_head   = 0;
    link_rx_tail   = 0;
    link_isr_count = 0;
    link_tick      = 0;
    link_cmd_len   = 0;
    link_last_seq  = 0;
    link_keys      = 0;
    link_hold      = 0;
    link_cmd_flags = 0;
    link_joy_mask  = 0xFFu;
    link_timeout   = LINK_TIMEOUT_TICKS;
    link_active    = 0;
    link_polls     = 0;
    link_rx_errors = 0;
    link_lost_irq  = 0;
    link_restoring = 0;

    // Clear any stale serial completion, then enable the serial interrupt.
    // (set_interrupts() also writes IF = 0.)
    set_interrupts(IE_REG | SIO_IFLAG);

#ifdef LINK_EXT_CLOCK
    // Slave: pre-load an idle byte and wait for the peer's clock.
    link_busy = 1;
    SB_REG = LINK_IDLE_BYTE;
    SC_REG = LINK_SC_ARM;
#else
    link_busy = 0;
#endif
}


// Called once per input poll by UPDATE_KEYS() (input.h)
uint8_t link_merge_keys(uint8_t joy) {
    uint8_t k = link_keys;
    link_polls++;
    if ((link_cmd_flags & LINK_CMDF_HOLD_POLLS) && link_hold) {
        link_hold--;
        if (link_hold == 0) link_keys = 0;  // released for the following poll
    }
    return (joy & link_joy_mask) | k;
}


static void link_apply_cmd(void) {
    // link_cmd_buf holds a checksum-valid packet
    link_timeout = 0;
    if ((!link_active) || (link_cmd_buf[1] != link_last_seq)) {
        // New command (idempotence: a repeated seq is only a keep-alive)
        link_last_seq  = link_cmd_buf[1];
        link_keys      = link_cmd_buf[2];
        link_hold      = link_cmd_buf[3];
        link_cmd_flags = link_cmd_buf[4];
        link_joy_mask  = (link_cmd_flags & LINK_CMDF_MASK_JOYPAD) ? 0x00u : 0xFFu;
    }
    link_active = 1;
}


static void link_parse_byte(uint8_t b) {

    if (link_cmd_len == 0) {
        if (b != LINK_CMD_SYNC) return;  // hunting for sync (0xFF = no peer, 0x00 = idle)
    }
    link_cmd_buf[link_cmd_len++] = b;

    if (link_cmd_len == LINK_CMD_LEN) {
        uint8_t a = 0, s = 0, c;
        for (c = 0; c < (LINK_CMD_LEN - 2u); c++) {
            a += link_cmd_buf[c];
            s += a;
        }
        if ((a == link_cmd_buf[LINK_CMD_LEN - 2u]) && (s == link_cmd_buf[LINK_CMD_LEN - 1u])) {
            link_apply_cmd();
            link_cmd_len = 0;
        } else {
            // Bad packet: resync on the next sync byte inside what we already have
            link_rx_errors++;
            uint8_t n = 0;
            for (c = 1; c < LINK_CMD_LEN; c++) {
                if (n) link_cmd_buf[n++] = link_cmd_buf[c];
                else if (link_cmd_buf[c] == LINK_CMD_SYNC) { link_cmd_buf[0] = LINK_CMD_SYNC; n = 1; }
            }
            link_cmd_len = n;
        }
    }
}


static void link_build_state(void) {
    uint8_t * p = link_tx_buf;
    uint8_t flags = 0;
    uint8_t ship_state = state.ship_state;
    uint8_t idx, cnt;

    if (state.paused)                              flags |= LINK_SF_PAUSED;
    if (ship_state & SHIP_BITS_PLAYER_ACTIVE)      flags |= LINK_SF_PLAYER_ACTIVE;
    if (ship_state == SHIP_STATE_CRASHING)         flags |= LINK_SF_CRASHING;
    if (ship_state == SHIP_STATE_CRASH_ENDED)      flags |= LINK_SF_CRASH_ENDED;
    if (game_state == GAME_STATE_OVER)             flags |= LINK_SF_GAME_OVER;
    if (link_polls)                                flags |= LINK_SF_INPUT_POLLED;
    if (link_active)                               flags |= LINK_SF_LINK_ACTIVE;
    if (link_restoring)                            flags |= LINK_SF_RESTORING;
    link_polls = 0;

    *p++ = LINK_STATE_SYNC;                        //  0
    *p++ = (uint8_t)link_tick;                     //  1 tick lo
    *p++ = (uint8_t)(link_tick >> 8);              //  2 tick hi
    *p++ = game_state;                             //  3 mode
    *p++ = flags;                                  //  4
    *p++ = ship_state;                             //  5
    *p++ = state.ship_counter;                     //  6
    *p++ = state.ship_x.h;                         //  7
    *p++ = state.ship_x.l;                         //  8
    *p++ = state.ship_y.h;                         //  9
    *p++ = state.ship_y.l;                         // 10
    *p++ = state.ship_z.h;                         // 11
    *p++ = state.ship_z.l;                         // 12
    *p++ = (uint8_t)lives_count;                   // 13 lives (BCD)
    *p++ = ((uint8_t *)&state.score)[0];           // 14 score BCD, least significant byte first
    *p++ = ((uint8_t *)&state.score)[1];           // 15
    *p++ = ((uint8_t *)&state.score)[2];           // 16
    *p++ = state.game_level;                       // 17
    *p++ = (uint8_t)(state.cur_level.obst_speed >> 2); // 18 obstacle speed, 1/64 px per frame
    // 19 canyon left edge x at ship centre row (same expression as entity_collisions.c)
    *p++ = CANYON_LEFT_X_BASE - state.p_scx_table_frame_base[state.ship_y.h + (sprite_ship_HEIGHT / 2u)];
    cnt = state.obstacles_active_count;
    *p++ = cnt;                                    // 20
    // 21..26: two oldest active obstacles (= lowest on screen = nearest the ship)
    idx = state.obstacles_active_first;
    for (uint8_t c = 0; c < 2u; c++) {
        if (c < cnt) {
            uint8_t y = state.obstacles[idx].y.h;
            *p++ = y;                                                  // y (top edge)
            *p++ = CANYON_LEFT_X_BASE - state.p_scx_table_frame_base[y]; // x (left edge of 32px sprite)
            *p++ = state.obstacles[idx].type;                          // type (bit7 = hidden)
            idx++;
            if (idx == OBSTACLES_COUNT_WRAP) idx = 0;
        } else {
            *p++ = 0xFFu; *p++ = 0xFFu; *p++ = 0xFFu;
        }
    }
    *p++ = link_last_seq;                          // 27 ack: seq of last applied command

    // 28, 29: Fletcher-style checksum over bytes 0..27 (mod 256)
    {
        uint8_t a = 0, s = 0;
        uint8_t * q = link_tx_buf;
        while (q != p) {
            a += *q++;
            s += a;
        }
        *p++ = a;
        *p   = s;
    }
}


void link_frame(void) {
    uint8_t c, head;

    link_tick++;

    // ---- Watchdog: keep the serial IRQ enabled and replay a lost completion.
    if (!(IE_REG & SIO_IFLAG)) IE_REG |= SIO_IFLAG;
    c = link_isr_count;
    if (link_busy && !(SC_REG & SIOF_XFER_START) && !(IF_REG & SIO_IFLAG) && (c == link_isr_count)) {
        // Transfer finished but its interrupt was cleared before it could run
        link_lost_irq++;
        link_replay_isr();
    }

    // ---- Hold countdown in frame (tick) units
    if (!(link_cmd_flags & LINK_CMDF_HOLD_POLLS) && link_hold) {
        link_hold--;
        if (link_hold == 0) link_keys = 0;
    }

    // ---- Parse received bytes
    head = link_rx_head;
    while (link_rx_tail != head) {
        link_parse_byte(link_rx_ring[link_rx_tail]);
        link_rx_tail = (link_rx_tail + 1u) & LINK_RX_MASK;
    }

    // ---- Timeout -> fall back to joypad only
    if (link_timeout < 0xFFu) link_timeout++;
    if (link_timeout > LINK_TIMEOUT_TICKS) {
        if (link_active) {
            link_active   = 0;
            link_keys     = 0;
            link_hold     = 0;
            link_joy_mask = 0xFFu;
        }
    }

    // ---- Publish a new state snapshot once the previous packet is fully out
#ifdef LINK_EXT_CLOCK
    if (link_tx_pos >= LINK_STATE_LEN) {
        link_build_state();
        link_tx_pos = 0;   // ISR starts pre-loading from byte 0 at the next exchange
    }
#else
    if ((!link_busy) && (link_tx_pos >= LINK_STATE_LEN)) {
        link_build_state();
        link_busy   = 1;
        link_tx_pos = 1;
        SB_REG = link_tx_buf[0];
        SC_REG = LINK_SC_ARM;
    }
#endif
}

#endif // LINK_INPUT
