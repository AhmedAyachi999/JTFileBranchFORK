import struct

from shape.topoMeshCompressedLODData import TopoMeshCompressedLODData
from shape.topoMeshLODData import TopoMeshLODData
from shape.topologicallyCompressedRepData import TopologicallyCompressedRepData
from lsg.types import JtVersion


class TopoMeshTopologicallyCompressedLODData:
    """
    7.2.2.1.2.4 TopoMesh Topologically Compressed LOD Data

    TopoMesh Topologically Compressed LOD Data collection contains the common items to all TopoMesh Topologically
    Compressed LOD data elements.
    """
    topo_mesh_lod_data: TopoMeshLODData
    version_number: int
    topo_mesh_compressed_rep_data: TopologicallyCompressedRepData

    @classmethod
    def from_bytes(cls, e_bytes, version=JtVersion.V9d5, end_offset=None):
        preview_len = 64 if e_bytes.remaining() >= 64 else e_bytes.remaining()
        preview = e_bytes.bytes[e_bytes.offset:e_bytes.offset + preview_len].hex(" ")
        topo_mesh_lod_data = TopoMeshLODData.from_bytes(
            e_bytes, version=version)
        if version == JtVersion.V9d5:
            version_number = struct.unpack("<h", e_bytes.read(2))[0]
        else:
            version_number = struct.unpack("<B", e_bytes.read(1))[0]

        if version_number != 1:
            RuntimeError(
                f"version {version_number} not supported in {__name__}")
        topo_mesh_compressed_rep_data = TopologicallyCompressedRepData.from_bytes(
            e_bytes, version=version, end_offset=end_offset)

        preview_len = 64 if e_bytes.remaining() >= 64 else e_bytes.remaining()
        preview = e_bytes.bytes[e_bytes.offset:e_bytes.offset + preview_len].hex(" ")
        return TopoMeshCompressedLODData(topo_mesh_lod_data, version_number, topo_mesh_compressed_rep_data)
