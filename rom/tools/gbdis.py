import sys, re
from pyboy.core.opcodes import CPU_COMMANDS
rom = open(sys.argv[1],'rb').read()
start = int(sys.argv[2],16); end = int(sys.argv[3],16)
def length(name):
    if 'd16' in name or 'a16' in name: return 3
    if 'd8' in name or 'r8' in name or 'a8' in name: return 2
    return 1
pc = start
while pc < end:
    op = rom[pc]
    if op == 0xCB:
        name = CPU_COMMANDS[0x100 + rom[pc+1]]; n = 2
    else:
        name = CPU_COMMANDS[op]; n = length(name)
    args = rom[pc+1:pc+n] if op != 0xCB else b''
    imm = ''
    if n == 3 and op != 0xCB: imm = ' ; $%04X' % (args[0] | args[1] << 8)
    elif n == 2 and op != 0xCB: imm = ' ; $%02X' % args[0]
    print('%04X: %-10s %s%s' % (pc, rom[pc:pc+n].hex(), name, imm))
    pc += n
