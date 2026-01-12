import logging
import struct
from dataclasses import dataclass

from codec.i32Cdp2 import I32CDP2, PredictorType
from shape.topologicallyCompressedVertexRecords import TopologicallyCompressedVertexRecords
from util import byteStream as bs
from util.jt_hash import jt_hash32
from lsg.types import JtVersion

logger = logging.getLogger(__name__)


@dataclass
class TopologicallyCompressedRepData:
    """
    7.2.2.1.2.5 Topologically Compressed Rep Data

    JT v9 represents triangle strip data very differently than it does in the JT v8 format. The new scheme stores
    the triangles from a TriStripSet as a topologically-connected triangle mesh. Even though more information is
    stored to the JT file, the additional structure provided by storing the full topological adjacency information
    actually provides a handsome reduction in the number of bytes needed to encode the triangles. More importantly,
    however, the topological information aids us in a more significant respect -- that of only storing the unique
    vertex records used by the TriStripSet. Combined, these two effects reduce the typical storage footprint of
    TriStripSet data by approximately half relative to the JT v8 format.

    The tristrip information itself is no longer stored in the JT file -- only the triangles themselves. The
    reader is expected to re-tristrip (or not) as it sees fit, as tristrips may no longer provide a performance
    advantage during rendering. There may, however, remain some memory savings for tristripping, and so the decision
    to tristrip is left to the user.

    To begin the decoding process, first read the compressed data fields shown in Figure 89. These fields provide
    all the information necessary to reconstruct the per face-group organized sets of triangles. The first 22 fields
    represent the topological information, and the remaining fields constitute the set of unique vertex records to be
    used. The next step is to run the topological decoder algorithm detailed in Appendix E: Polygon Mesh Topology
    Coder on this data to reconstruct the topologically connected representation of the triangle mesh in a so-called
    "dual VFMesh.' The triangles in this heavy-weight data structure can then be exported to a lighter-weight form,
    and the dual VFMesh discarded if desired.
    """

    face_degrees: list
    vertex_valences: list
    vertex_groups: list
    vertex_flags: list
    face_attribute_masks: list
    face_attribute_masks8_30: list
    face_attribute_masks8_4: list
    high_degree_face_attribute_mask: list
    split_face_syms: list
    split_face_positions: list
    hash: int
    topologically_compressed_vertex_records: TopologicallyCompressedVertexRecords

    @classmethod
    def from_bytes(cls, e_bytes, version=JtVersion.V9d5):
        logger.debug(f'creating from bytes')
        logger.debug(
            (e_bytes.bytes[e_bytes.offset:e_bytes.offset+30]).hex(" "))
        face_degrees = []
        for i in range(8):
            face_degrees.append(I32CDP2.read_vec_i_32(e_bytes))
        vertex_valences = I32CDP2.read_vec_i_32(e_bytes)
        vertex_groups = I32CDP2.read_vec_i_32(e_bytes)
        vertex_flags = I32CDP2.read_vec_i_32(e_bytes, PredictorType.PredLag1)

        face_attribute_masks = []
        for i in range(8):
            face_attribute_masks.append(I32CDP2.read_vec_i_32(e_bytes))

        face_attribute_masks8_30 = I32CDP2.read_vec_i_32(e_bytes)
        face_attribute_masks8_4 = I32CDP2.read_vec_i_32(e_bytes)
        high_degree_face_attribute_mask = bs.read_vec_i_32(e_bytes)
        # Here is the problem. Is it due to the previous readings ?
        split_face_syms = I32CDP2.read_vec_i_32(
            e_bytes, PredictorType.PredLag1)
        # Here is the problem. Is it due to the previous readings ?
        split_face_positions = I32CDP2.read_vec_i_32(e_bytes)

        read_hash = struct.unpack("<I", e_bytes.read(4))[0]

        topologically_compressed_vertex_records = TopologicallyCompressedVertexRecords.from_bytes(
            e_bytes)
        return TopologicallyCompressedRepData(face_degrees,
                                              vertex_valences,
                                              vertex_groups,
                                              vertex_flags,
                                              face_attribute_masks,
                                              face_attribute_masks8_30,
                                              face_attribute_masks8_4,
                                              high_degree_face_attribute_mask,
                                              split_face_syms,
                                              split_face_positions,
                                              read_hash,
                                              topologically_compressed_vertex_records)

    @classmethod
    def compute_hash(cls, face_attribute_masks, face_attribute_masks8_30, face_attribute_masks8_4, face_degrees,
                     high_degree_face_attribute_mask, split_face_positions, split_face_syms, vertex_flags,
                     vertex_groups, vertex_valences):
        comp_hash = 0
        for fd in face_degrees[:-1]:
            comp_hash = jt_hash32(fd, comp_hash)
        comp_hash = jt_hash32(vertex_valences, comp_hash)
        comp_hash = jt_hash32(vertex_groups, comp_hash)
        comp_hash = jt_hash32(vertex_flags, comp_hash)
        for am in face_attribute_masks:
            comp_hash = jt_hash32(am, comp_hash)
        comp_hash = jt_hash32(face_attribute_masks[7], comp_hash)
        comp_hash = jt_hash32(face_attribute_masks8_30, comp_hash)
        comp_hash = jt_hash32(face_attribute_masks8_4, comp_hash)
        comp_hash = jt_hash32(high_degree_face_attribute_mask, comp_hash)
        comp_hash = jt_hash32(split_face_syms, comp_hash)
        comp_hash = jt_hash32(split_face_positions, comp_hash)
        return comp_hash
