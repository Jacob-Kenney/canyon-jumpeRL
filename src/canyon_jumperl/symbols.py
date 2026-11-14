"""Resolve RAM addresses / struct offsets from the GBDK linker outputs.

* ``.noi``  -> ``DEF <symbol> <hex addr>``  (global symbol addresses)
* ``.cdb``  -> SDCC debug info; ``T:F<file>$<struct>[({off}S:S$field$...({size}...)...]``
               gives byte offsets of struct fields (e.g. ``game_state_data``).

Nothing about the ``state`` layout is hardcoded: it is parsed from the .cdb.
C globals carry a leading underscore in the .noi (``_state``).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_NOI_RE = re.compile(r"^DEF\s+(\S+)\s+0x([0-9A-Fa-f]+)\s*$")
_FIELD_RE = re.compile(r"\(\{(\d+)\}S:S\$(\w+)\$0_0\$0\(\{(\d+)\}")


@dataclass(frozen=True)
class Field:
    offset: int
    size: int


class Symbols:
    def __init__(self, noi_path: str | Path, cdb_path: str | Path | None = None):
        noi_path = Path(noi_path)
        cdb_path = Path(cdb_path) if cdb_path else noi_path.with_suffix(".cdb")
        self.globals: dict[str, int] = {}
        for line in noi_path.read_text().splitlines():
            m = _NOI_RE.match(line)
            if m:
                self.globals[m.group(1)] = int(m.group(2), 16)
        self._structs: dict[str, dict[str, Field]] = {}
        for line in cdb_path.read_text().splitlines():
            if not line.startswith("T:F"):
                continue
            head, _, body = line.partition("[")
            name = head.rsplit("$", 1)[-1]
            fields = {m.group(2): Field(int(m.group(1)), int(m.group(3)))
                      for m in _FIELD_RE.finditer(body)}
            # the same struct is emitted once per translation unit; keep the first
            self._structs.setdefault(name, fields)

    def addr(self, c_name: str) -> int:
        """Address of C global ``c_name`` (e.g. ``state``, ``lives_count``, ``_rand_seed``)."""
        key = "_" + c_name
        if key not in self.globals:
            raise KeyError(f"symbol {c_name!r} not found in .noi")
        return self.globals[key]

    def field(self, struct: str, name: str) -> Field:
        try:
            return self._structs[struct][name]
        except KeyError:
            raise KeyError(f"field {struct}.{name} not found in .cdb") from None

    def struct_size(self, struct: str) -> int:
        return max(f.offset + f.size for f in self._structs[struct].values())
