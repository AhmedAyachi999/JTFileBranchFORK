import struct
from dataclasses import dataclass

from codec.i32Cdp2 import I32CDP2


@dataclass
class TopoMeshCompressedRepData:
    """
    7.2.2.1.2.8 TopoMesh Compressed Rep Data V2

    TopoMesh Compressed Rep Data V2 data contains additional geometric shape data (auxiliary vertex fields) that were
    not included in V1. Auxiliary fields are parallel to the existing vertex record information and contain additional
    information pertaining to each vertex.
    """

    @classmethod
    def from_bytes(cls, e_bytes):
        preview_len = 128 if e_bytes.remaining() >= 128 else e_bytes.remaining()
        preview = e_bytes.bytes[e_bytes.offset:e_bytes.offset + preview_len].hex(" ")
        print(
            f"TopoMeshCompressedRepData.from_bytes entry offset={e_bytes.offset} "
            f"remaining={e_bytes.remaining()} next{preview_len}={preview}"
        )
        # Skip/consume fields for TopoMesh Compressed Rep Data V2.
        version_number = struct.unpack("<h", e_bytes.read(2))[0]
        if version_number != 1:
            raise RuntimeError(
                f"Version {version_number} not supported for {cls.__name__}"
            )
        _vertex_bindings = struct.unpack("<Q", e_bytes.read(8))[0]
        _aux_data_hash = struct.unpack("<i", e_bytes.read(4))[0]

        # Three VecU32{Int32CDP2} streams: lower mantissae, upper mantissae, exponents.
        _ = I32CDP2.read_vec_i_32(e_bytes)
        _ = I32CDP2.read_vec_i_32(e_bytes)
        _ = I32CDP2.read_vec_i_32(e_bytes)

        # Spec lists a subsequent I16 version number; read and ignore if present.
        _ = struct.unpack("<h", e_bytes.read(2))[0]

        return TopoMeshCompressedRepData()
