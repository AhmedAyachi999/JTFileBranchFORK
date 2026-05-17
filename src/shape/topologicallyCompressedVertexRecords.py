import struct
from dataclasses import dataclass, field

from lsg.types import JtVersion
from lsg.types import QuantizationParameters
from shape.compressedVertexColorArray import CompressedVertexColorArray
from shape.compressedVertexCoordinateArray import CompressedVertexCoordinateArray
from shape.compressedVertexFlagArray import CompressedVertexFlagArray
from shape.compressedVertexNormalArray import CompressedVertexNormalArray
from shape.compressedVertexTextureCoordinateArray import CompressedVertexTextureCoordinateArray


@dataclass
class TopologicallyCompressedVertexRecords:
    vertex_binding: int
    quantization_parameters: QuantizationParameters
    number_of_topological_vertices: int
    compressed_vertex_coordinate_array: CompressedVertexCoordinateArray = field(
        default=None)
    compressed_vertex_normal_array: CompressedVertexNormalArray = field(
        default=None)
    compressed_vertex_color_array: CompressedVertexColorArray = field(
        default=None)
    compressed_vertex_texture_coordinate_arrays: [
        CompressedVertexTextureCoordinateArray] = field(default_factory=list)
    compressed_vertex_flag_array: CompressedVertexFlagArray = field(
        default=None)

    @classmethod
    def from_bytes(cls, e_bytes, version=JtVersion.V9d5, coordinates_only=False):
        vertex_binding = struct.unpack("<Q", e_bytes.read(8))[0]
        quantization_parameters = QuantizationParameters.from_bytes(e_bytes)
        number_of_topological_vertices = struct.unpack(
            "<i", e_bytes.read(4))[0]
        if number_of_topological_vertices <= 0:
            return TopologicallyCompressedVertexRecords(vertex_binding, quantization_parameters,
                                                        number_of_topological_vertices)

        number_of_vertex_attributes = struct.unpack("<i", e_bytes.read(4))[0]
        compressed_vertex_coordinate_array = None
        if (vertex_binding & 0x07) != 0:
            compressed_vertex_coordinate_array = CompressedVertexCoordinateArray.from_bytes(
                e_bytes, version=version)
        if coordinates_only:
            return TopologicallyCompressedVertexRecords(
                vertex_binding,
                quantization_parameters,
                number_of_topological_vertices,
                compressed_vertex_coordinate_array,
                None,
                None,
                [None] * 8,
                None,
            )
        compressed_vertex_normal_array = None
        if (vertex_binding & 0x08) != 0:
            compressed_vertex_normal_array = CompressedVertexNormalArray.from_bytes(
                e_bytes, version=version)
        compressed_vertex_color_array = None
        if (vertex_binding & 0x30) != 0:
            compressed_vertex_color_array = CompressedVertexColorArray.from_bytes(
                e_bytes)
        compressed_vertex_texture_coordinate_arrays = [None] * 8
        if (vertex_binding & 0xf00) != 0:
            compressed_vertex_texture_coordinate_arrays[0] = CompressedVertexTextureCoordinateArray.from_bytes(
                e_bytes)
        if (vertex_binding & 0xf000) != 0:
            compressed_vertex_texture_coordinate_arrays[1] = CompressedVertexTextureCoordinateArray.from_bytes(
                e_bytes)
        if (vertex_binding & 0xf0000) != 0:
            compressed_vertex_texture_coordinate_arrays[2] = CompressedVertexTextureCoordinateArray.from_bytes(
                e_bytes)
        if (vertex_binding & 0xf00000) != 0:
            compressed_vertex_texture_coordinate_arrays[3] = CompressedVertexTextureCoordinateArray.from_bytes(
                e_bytes)
        if (vertex_binding & 0xf000000) != 0:
            compressed_vertex_texture_coordinate_arrays[4] = CompressedVertexTextureCoordinateArray.from_bytes(
                e_bytes)
        if (vertex_binding & 0xf0000000) != 0:
            compressed_vertex_texture_coordinate_arrays[5] = CompressedVertexTextureCoordinateArray.from_bytes(
                e_bytes)
        if (vertex_binding & 0xf00000000) != 0:
            compressed_vertex_texture_coordinate_arrays[6] = CompressedVertexTextureCoordinateArray.from_bytes(
                e_bytes)
        if (vertex_binding & 0xf000000000) != 0:
            compressed_vertex_texture_coordinate_arrays[7] = CompressedVertexTextureCoordinateArray.from_bytes(
                e_bytes)

        compressed_vertex_flag_array = None
        if (vertex_binding & 0x40) != 0:
            compressed_vertex_flag_array = CompressedVertexFlagArray.from_bytes(
                e_bytes, version=version)
        return TopologicallyCompressedVertexRecords(vertex_binding,
                                                    quantization_parameters,
                                                    number_of_topological_vertices,
                                                    compressed_vertex_coordinate_array,
                                                    compressed_vertex_normal_array,
                                                    compressed_vertex_color_array,
                                                    compressed_vertex_texture_coordinate_arrays,
                                                    compressed_vertex_flag_array)
