import struct
from dataclasses import dataclass

from codec.i32Cdp2 import I32CDP2, PredictorType
from shape.quantizer import PointQuantizerData


@dataclass
class CompressedVertexCoordinateArray:
    unique_vertex_count: int
    number_components: int
    point_quantizer_data: PointQuantizerData
    vertex_coordinates_expnonents: [[int]]
    vertex_coordinates_mantissae: [[int]]
    vertex_coordinates_codes: [[int]]
    vertex_coordinate_hash: int
    vertex_coordinates: [[float]]

    @classmethod
    def from_bytes(cls, e_bytes):
        unique_vertex_count, number_components = struct.unpack(
            "<iB", e_bytes.read(5))
        point_quantizer_data = PointQuantizerData.from_bytes(e_bytes)
        vertex_exponents = []
        vertex_mantissae = []
        vertex_codes = []
        vertex_coordinates = []
        if point_quantizer_data.number_of_bits() == 0:
            for i in range(number_components):
                exponents = (I32CDP2.read_vec_i_32(
                    e_bytes, PredictorType.PredLag1))
                mantissae = (I32CDP2.read_vec_i_32(
                    e_bytes, PredictorType.PredLag1))
                codes = [(e << 23 | m) & 0xffffffff for e,
                         m in zip(exponents, mantissae)]

                vertex_exponents.append(exponents)
                vertex_mantissae.append(mantissae)
                vertex_codes.append(codes)
        else:
            for i in range(number_components):
                vertex_codes.append(I32CDP2.read_vec_i_32(
                    e_bytes, PredictorType.PredLag1))

        if point_quantizer_data.number_of_bits() == 0:
            for codes in vertex_codes:
                coordinates = [struct.unpack("f", c.to_bytes(4, "little"))[
                    0] for c in codes]
                vertex_coordinates.append(coordinates)
        else:
            quantizers = [
                point_quantizer_data.x,
                point_quantizer_data.y,
                point_quantizer_data.z,
            ]
            for component_index, codes in enumerate(vertex_codes):
                if component_index < len(quantizers):
                    quantizer = quantizers[component_index]
                else:
                    quantizer = quantizers[-1]
                vertex_coordinates.append(
                    _dequantize_codes(codes, quantizer.min_v, quantizer.max_v, quantizer.number_of_bits)
                )
        vertex_coordinate_hash = struct.unpack("<i", e_bytes.read(4))[0]
        return CompressedVertexCoordinateArray(unique_vertex_count,
                                               number_components,
                                               point_quantizer_data,
                                               vertex_exponents,
                                               vertex_mantissae,
                                               vertex_codes,
                                               vertex_coordinate_hash,
                                               vertex_coordinates)


def _dequantize_codes(codes, min_value: float, max_value: float, bits: int) -> list[float]:
    if bits <= 0:
        return [float(code) for code in codes]

    max_code = (1 << bits) - 1
    if max_code <= 0:
        return [float(min_value) for _ in codes]

    span = max_value - min_value
    if span == 0:
        return [float(min_value) for _ in codes]

    return [
        float(min_value + (int(code) / max_code) * span)
        for code in codes
    ]
