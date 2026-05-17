import struct
from dataclasses import dataclass

from codec.i32Cdp2 import I32CDP2
from lsg.types import JtVersion


@dataclass
class CompressedVertexFlagArray:
    vertex_flag_count: int
    vertex_flags: list

    @classmethod
    def from_bytes(cls, e_bytes, version=JtVersion.V9d5):
        vertex_flag_count = struct.unpack("<I", e_bytes.read(4))[0]
        vertex_flags = I32CDP2.read_vec_i_32(e_bytes, version=version)
        return CompressedVertexFlagArray(vertex_flag_count, vertex_flags)
