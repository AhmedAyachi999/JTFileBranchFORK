import struct
from dataclasses import dataclass

from codec.deeringNormal import DeeringNormalCodec
from codec.i32Cdp2 import I32CDP2
from lsg.types import JtVersion


@dataclass
class CompressedVertexNormalArray:
    normal_count: int
    number_components: int
    vertex_normal_hash: int
    normal_coordinates: [[int]]

    @classmethod
    def from_bytes(cls, e_bytes, version=JtVersion.V9d5):
        normal_count, number_components, quantization_bits = struct.unpack(
            "<iBB", e_bytes.read(6))
        if quantization_bits == 0:
            vertex_normal_exponents = []
            vertex_normal_mantissae = []
            vertex_normal_codes = []
            vertex_normal_coordinates = []
            for i in range(number_components):
                normal_exponents = I32CDP2.read_vec_i_32(e_bytes, version=version)
                normal_mantissae = I32CDP2.read_vec_i_32(e_bytes, version=version)
                normal_code = [(e << 23 | m) & 0xffffffff for e,
                               m in zip(normal_exponents, normal_mantissae)]

                vertex_normal_exponents.append(normal_exponents)
                vertex_normal_mantissae.append(normal_mantissae)
                vertex_normal_codes.append(normal_code)

            for codes in vertex_normal_codes:
                coordinates = [struct.unpack("f", c.to_bytes(4, "little"))[
                    0] for c in codes]
                vertex_normal_coordinates.append(coordinates)
        else:
            sextant_codes = I32CDP2.read_vec_i_32(e_bytes, version=version)
            octant_codes = I32CDP2.read_vec_i_32(e_bytes, version=version)
            theta_codes = I32CDP2.read_vec_i_32(e_bytes, version=version)
            psi_codes = I32CDP2.read_vec_i_32(e_bytes, version=version)
            dnc = DeeringNormalCodec(quantization_bits)
            xyz = [dnc.convert_code_to_vec(s, o, t, p) for s, o, t, p in
                   zip(sextant_codes, octant_codes, theta_codes, psi_codes)]
            vertex_normal_coordinates = [[p[0] for p in xyz], [
                p[1] for p in xyz], [p[2] for p in xyz]]

        vertex_normal_hash = struct.unpack("<I", e_bytes.read(4))[0]
        return CompressedVertexNormalArray(normal_count,
                                           number_components,
                                           vertex_normal_hash,
                                           vertex_normal_coordinates)
