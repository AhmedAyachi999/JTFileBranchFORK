import struct
from dataclasses import dataclass

from shape.topoMeshCompressedLODData import TopoMeshCompressedLODData
from shape.topoMeshTopologicallyCompressedLODData import TopoMeshTopologicallyCompressedLODData
from lsg.types import JtVersion
from lsg.elementHeader import ElementHeader
import logging

logger = logging.getLogger(__name__)


@dataclass
class VertexShapeLODData:
    """
    7.2.2.1.2.1 Vertex Shape LOD Data

    Vertex Shape LOD Data collection is an abstract container for geometric primitives such as triangle strips, line
    strips, or points, depending on the specific type of Vertex Shape. The set of primitives are further partitioned
    into so-called "face groups." The Vertex Shape LOD Data also contains the vertex attribute bindings and quantization
    settings used to store the vertex records referenced by the primitives.

    One use for face groups is to establish a correspondence between Brep faces and their triangle representation.
    A convention for mapping JTBrep and XTBrep faces to face groups is described in section 9.10 Brep Face Group
    Associations.
    """
    version_number: int
    vertex_bindings: int
    topo_mesh_compressed_lod_data: TopoMeshCompressedLODData

    @classmethod
    def from_bytes(cls, e_bytes, shape, version=JtVersion.unsupported):
        if version == JtVersion.V9d5:
            version_number = struct.unpack("<h", e_bytes.read(2))[0]
        elif version == JtVersion.unsupported:
            logger.critical("unsupported jt version")
            raise RuntimeError("unsupported jt version")
        else:
            e_bytes.read(1)  # base shape lod data
            version_number = struct.unpack("B", e_bytes.read(1))[0]
        if version_number != 1:
            raise RuntimeError(
                f"Version {version_number} not supported for {cls.__name__}")
        vertex_binding = struct.unpack("<Q", e_bytes.read(8))[0]
        """
        Bits  1-3:
            Vertex Coordinate Binding. The Vertex Coordinate Binding denotes per vertex coordinate field 
            data is present when one of the bits is set.

            Bit 1: 2 Component Vertex Coordinates,
            Bit 2: 3 Component Vertex Coordinates,
            Bit 3: 4 Component Vertex Coordinates,
        Bit 4:
            Normal Binding. The Normal Binding denotes per vertex normal field data is present when the bit 
            is set. Normal field data is always stored in 3 Component Normals when present.
        Bits 5-6:
            Color Binding. The Color Binding denotes per vertex color field data is present when one of the 
            bits is set.

            Bit 5: 3 Component Colors,
            Bit 6: 4 Component Color
        Bit 7:
            Vertex Flag Binding. The Vertex Flag Binding denotes the per vertex flag field is present
            when one of the bits is set.
        Bits 9-12:
            Texture Coordinate 0 Binding. The Texture Coordinate 0 binding denotes per vertex texture 
            coordinates field data is present when one of the bits is set:

            Bit  9: 1 Component Texture Coordinates
            Bit 10: 2 Component Texture Coordinates
            Bit 11: 3 Component Texture Coordinates
            Bit 12: 4 Component Texture Coordinates 
        Bits 13-16:
            Texture Coordinate 1 Binding. The Texture Coordinate 1 binding denotes per vertex texture coordinates field 
            data is present when one of the bits is set:

            Bit 13: 1 Component Texture Coordinates
            Bit 14: 2 Component Texture Coordinates
            Bit 15: 3 Component Texture Coordinates
            Bit 16: 4 Component Texture Coordinates
        Bits 17-20:
            Texture Coordinate 2 Binding. The Texture Coordinate 2 binding denotes per vertex texture coordinates field 
            data is present when one of the bits is set:

            Bit 17: 1 Component Texture Coordinates
            Bit 18: 2 Component Texture Coordinates
            Bit 19: 3 Component Texture Coordinates
            Bit 20: 4 Component Texture Coordinates
        Bits 21-24:
            Texture Coordinate 3 Binding. The Texture Coordinate 3 binding denotes per vertex texture coordinates field 
            data is present when one of the bits is set:

            Bit 21: 1 Component Texture Coordinates
            Bit 22: 2 Component Texture Coordinates
            Bit 23: 3 Component Texture Coordinates
            Bit 24: 4 Component Texture Coordinates
        Bits 25-28:
            Texture Coordinate 4 Binding. The Texture Coordinate 4 binding denotes per vertex texture coordinates field 
            data is present when one of the bits is set:

            Bit 25: 1 Component Texture Coordinates
            Bit 26: 2 Component Texture Coordinates
            Bit 27: 3 Component Texture Coordinates
            Bit 28: 4 Component Texture Coordinates
        Bits 29-32:
            Texture Coordinate 5 Binding. The Texture Coordinate 5 binding denotes per vertex texture coordinates field 
            data is present when one of the bits is set:

            Bit 29: 1 Component Texture Coordinates
            Bit 30: 2 Component Texture Coordinates
            Bit 31: 3 Component Texture Coordinates
            Bit 32: 4 Component Texture Coordinates
        Bits 33-36:
            Texture Coordinate 6 Binding. The Texture Coordinate 6 binding denotes per vertex texture coordinates field 
            data is present when one of the bits is set:

            Bit 33: 1 Component Texture Coordinates
            Bit 34: 2 Component Texture Coordinates
            Bit 35: 3 Component Texture Coordinates
            Bit 36: 4 Component Texture Coordinates
        Bits 37-40:
            Texture Coordinate 7 Binding. The Texture Coordinate 7 binding denotes per vertex texture coordinates field 
            data is present when one of the bits is set:

            Bit 37: 1 Component Texture Coordinates
            Bit 38: 2 Component Texture Coordinates
            Bit 39: 3 Component Texture Coordinates
            Bit 40: 4 Component Texture Coordinates
        Bit 64:
            Auxiliary Vertex Field Binding. The Auxiliary Vertex Field Binding denotes per vertex auxiliary field data 
            is present on the shape when the bit is set.
        """

        nested_element_end = None
        if version == JtVersion.V10d5:
            nested_header = ElementHeader.from_bytes(e_bytes)
            nested_payload_length = max(0, nested_header.length - 25)
            nested_element_end = min(
                len(e_bytes.bytes), e_bytes.offset + nested_payload_length
            )
        elif version == JtVersion.V9d5:
            e_bytes.read(2)
        if shape == "Tri-Strip":
            topo_mesh_compressed_lod_data = TopoMeshTopologicallyCompressedLODData.from_bytes(
                e_bytes, version=version, end_offset=nested_element_end)
        else:
            e_bytes.read(2)
            topo_mesh_compressed_lod_data = TopoMeshCompressedLODData.from_bytes(
                e_bytes, version=version)
        return VertexShapeLODData(version_number, vertex_binding, topo_mesh_compressed_lod_data)
