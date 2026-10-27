"""Read PE64 loader metadata without executing Windows binaries (Microsoft PE/COFF)."""

import struct
from pathlib import Path


class PEImage:
    def __init__(self, data: bytes):
        self.data = data
        if len(data) < 64 or data[:2] != b"MZ":
            raise ValueError("Expected a Windows PE executable")
        pe = self.u32(0x3C)
        if data[pe : pe + 4] != b"PE\0\0":
            raise ValueError("Missing PE signature")
        self.machine, count = struct.unpack_from("<HH", data, pe + 4)
        self.timestamp = self.u32(pe + 8)
        optional_size = struct.unpack_from("<H", data, pe + 20)[0]
        optional = pe + 24
        if struct.unpack_from("<H", data, optional)[0] != 0x20B or not 1 <= count <= 96:
            raise ValueError("Only PE32+ with valid section count is supported")
        self.image_base = struct.unpack_from("<Q", data, optional + 24)[0]
        self.subsystem = struct.unpack_from("<H", data, optional + 68)[0]
        self.header_size = self.u32(optional + 60)
        directory_count = self.u32(optional + 108)
        if optional_size < 112 or directory_count > (optional_size - 112) // 8:
            raise ValueError("Invalid PE data directory bounds")
        self.directories = [
            struct.unpack_from("<II", data, optional + 112 + index * 8)
            for index in range(directory_count)
        ]
        self.sections = []
        for index in range(count):
            at = optional + optional_size + index * 40
            virtual_size, rva, raw_size, raw = struct.unpack_from("<4I", data, at + 8)
            self.sections.append((rva, max(virtual_size, raw_size), raw, raw_size))

    def u32(self, offset: int) -> int:
        return struct.unpack_from("<I", self.data, offset)[0]

    def offset(self, rva: int) -> int:
        if 0 <= rva < min(self.header_size, len(self.data)):
            return rva
        for start, size, raw, raw_size in self.sections:
            if start <= rva < start + size and rva - start < raw_size:
                value = raw + rva - start
                if value >= len(self.data):
                    break
                return value
        raise ValueError("PE RVA is outside file-backed sections")

    def string(self, rva: int) -> str:
        at = self.offset(rva)
        end = self.data.find(b"\0", at, at + 65536)
        if end < 0:
            raise ValueError("Unterminated PE string")
        return self.data[at:end].decode("ascii")

    def imports(self, *, delayed: bool = False) -> dict[str, list[str]]:
        directory = 13 if delayed else 1
        if len(self.directories) <= directory or not self.directories[directory][0]:
            return {}
        rva, _size = self.directories[directory]
        at, result = self.offset(rva), {}
        step = 32 if delayed else 20
        for _ in range(4096):
            if not any(self.data[at : at + step]):
                return result
            if delayed:
                if self.u32(at) != 1:
                    raise ValueError("Unexpected non-RVA delayed import descriptor")
                name, thunk = self.u32(at + 4), self.u32(at + 16)
            else:
                thunk, name = self.u32(at), self.u32(at + 12)
                thunk = thunk or self.u32(at + 16)
            symbols, pointer = [], self.offset(thunk)
            for _ in range(1_000_000):
                value = struct.unpack_from("<Q", self.data, pointer)[0]
                if not value:
                    break
                symbols.append(
                    f"#{value & 0xFFFF}" if value & (1 << 63) else self.string(value + 2)
                )
                pointer += 8
            else:
                raise ValueError("Excessive PE import table")
            result[self.string(name).lower()] = symbols
            at += step
        raise ValueError("Excessive PE import descriptors")

    def exports(self) -> dict[str, int]:
        if not self.directories or not self.directories[0][0]:
            return {}
        at = self.offset(self.directories[0][0])
        functions, names, ordinals = (self.offset(self.u32(at + i)) for i in (28, 32, 36))
        count = self.u32(at + 24)
        if count > 1_000_000:
            raise ValueError("Excessive PE export table")
        result = {}
        for index in range(count):
            name = self.string(self.u32(names + index * 4))
            ordinal = struct.unpack_from("<H", self.data, ordinals + index * 2)[0]
            result[name] = self.u32(functions + ordinal * 4)
        return result

    def to_dict(self) -> dict:
        return {
            "machine": hex(self.machine),
            "subsystem": self.subsystem,
            "timestamp": self.timestamp,
            "imports": self.imports(),
            "delayed_imports": self.imports(delayed=True),
            "exports": sorted(self.exports()),
        }


def inspect_pe(path: str | Path) -> dict:
    return PEImage(Path(path).read_bytes()).to_dict()
