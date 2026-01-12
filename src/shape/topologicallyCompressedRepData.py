import logging
import struct
import sys
from dataclasses import dataclass

from codec.i32Cdp2 import I32CDP2, PredictorType
from shape.topologicallyCompressedVertexRecords import TopologicallyCompressedVertexRecords
from util import byteStream as bs
from util.jt_hash import jt_hash16, jt_hash32_ints
from lsg.types import JtVersion
from shape.dual_vf_mesh import DualVFMesh
from typing import Optional, List

logger = logging.getLogger(__name__)

def _inc_mod(idx: int, mod: int) -> int:
    return (idx + 1) % mod if mod > 0 else 0


def _dec_mod(idx: int, mod: int) -> int:
    return (idx - 1 + mod) % mod if mod > 0 else 0
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
    used. The next step is to run the topological decoder algorithm detailed in Appendix E: tPolygon Mesh Topology
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
        print("creating from bytes")
        print((e_bytes.bytes[e_bytes.offset:e_bytes.offset+800]).hex(" "))

        def log_after(label: str):
            offset = e_bytes.offset
            remaining = e_bytes.remaining()
            preview_len = 50 if remaining > 50 else remaining
            preview = e_bytes.bytes[offset:offset + preview_len].hex(" ")
            print(f"after {label}: offset={offset} remaining={remaining} next50={preview}")

        def read_vec(label: str, predictor=PredictorType.PredNULL):
            print(f"read_vec_i_32 {label} start_offset={e_bytes.offset}")
            if label == "vertex_valences":
                start = e_bytes.offset
                if start + 5 <= len(e_bytes.bytes):
                    value_count = struct.unpack("<i", e_bytes.bytes[start:start + 4])[0]
                    codec_type = e_bytes.bytes[start + 4]
                    code_text_len = None
                    if codec_type in (1, 3) and start + 9 <= len(e_bytes.bytes):
                        code_text_len = struct.unpack("<i", e_bytes.bytes[start + 5:start + 9])[0]
                    print(
                        f"peek vertex_valences header: value_count={value_count} "
                        f"codec_type={codec_type} code_text_length={code_text_len}"
                    )
            e_bytes.debug_read = True
            out = I32CDP2.read_vec_i_32(e_bytes, predictor)
            print(out)
            print()
            e_bytes.debug_read = False
            print(f"read_vec_i_32 {label} end_offset={e_bytes.offset}")
            log_after(label)
            return out

        face_degrees = []
        for i in range(8):
            print(f"face_degrees[{i}] pre_offset={e_bytes.offset}")
            face_degrees.append(read_vec(f"face_degrees[{i}]"))
            print(f"face_degrees[{i}] post_offset={e_bytes.offset}")
        print(face_degrees)
        vertex_valences = read_vec("vertex_valences")
        vertex_groups = read_vec("vertex_groups")
        vertex_flags = read_vec("vertex_flags", PredictorType.PredLag1)

        face_attribute_masks = []
        for i in range(8):
            face_attribute_masks.append(read_vec(f"face_attribute_masks[{i}]"))

        face_attribute_masks8_30 = read_vec("face_attribute_masks8_30")
        face_attribute_masks8_4 = read_vec("face_attribute_masks8_4")
        # Experiment: decode high-degree masks using the same codec path.
        high_degree_face_attribute_mask = I32CDP2.read_vec_i_32(e_bytes)
        split_face_syms = read_vec("split_face_syms", PredictorType.PredLag1)
        split_face_positions = read_vec("split_face_positions")
        read_hash = struct.unpack("<I", e_bytes.read(4))[0]
        print(f"read composite hash (raw) = 0x{read_hash:08X}")
        computed_hash = cls.compute_hash(
            face_attribute_masks,
            face_attribute_masks8_30,
            face_attribute_masks8_4,
            face_degrees,
            high_degree_face_attribute_mask,
            split_face_positions,
            split_face_syms,
            vertex_flags,
            vertex_groups,
            vertex_valences,
        )
        if computed_hash != read_hash:
            logger.warning(
                "Composite hash mismatch: computed=0x%08X read=0x%08X",
                computed_hash,
                read_hash,
            )
            raise SystemExit(1)
        else:
            logger.info("Composite hash OK: 0x%08X", computed_hash)
            raise SystemExit(0)
        topologically_compressed_vertex_records = TopologicallyCompressedVertexRecords.from_bytes(
            e_bytes
        )
        print(
            "number_of_topological_vertices=",
            topologically_compressed_vertex_records.number_of_topological_vertices,
        )
        return TopologicallyCompressedRepData(
            face_degrees,
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
            topologically_compressed_vertex_records,
        )

    @classmethod
    def compute_hash(cls, face_attribute_masks, face_attribute_masks8_30, face_attribute_masks8_4, face_degrees,
                     high_degree_face_attribute_mask, split_face_positions, split_face_syms, vertex_flags,
                     vertex_groups, vertex_valences):
        comp_hash = 0
        face_degrees = face_degrees or []
        vertex_valences = vertex_valences or []
        vertex_groups = vertex_groups or []
        vertex_flags = vertex_flags or []
        face_attribute_masks = face_attribute_masks or []
        face_attribute_masks8_30 = face_attribute_masks8_30 or []
        face_attribute_masks8_4 = face_attribute_masks8_4 or []
        high_degree_face_attribute_mask = high_degree_face_attribute_mask or []
        split_face_syms = split_face_syms or []
        split_face_positions = split_face_positions or []

        for fd in face_degrees[:8]:
            comp_hash = jt_hash32_ints(fd or [], comp_hash)
        comp_hash = jt_hash32_ints(vertex_valences, comp_hash)
        comp_hash = jt_hash32_ints(vertex_groups, comp_hash)
        comp_hash = jt_hash16(vertex_flags, comp_hash)

        for i in range(min(7, len(face_attribute_masks))):
            comp_hash = jt_hash32_ints(face_attribute_masks[i] or [], comp_hash)
        # Context 7 is split into 30/30/4-bit chunks; assume they are stored separately.
        if len(face_attribute_masks) > 7:
            comp_hash = jt_hash32_ints(face_attribute_masks[7] or [], comp_hash)
        comp_hash = jt_hash32_ints(face_attribute_masks8_30, comp_hash)
        comp_hash = jt_hash32_ints(face_attribute_masks8_4, comp_hash)

        comp_hash = jt_hash32_ints(high_degree_face_attribute_mask, comp_hash)
        comp_hash = jt_hash32_ints(split_face_syms, comp_hash)
        comp_hash = jt_hash32_ints(split_face_positions, comp_hash)
        return comp_hash

# This class serves as a coordinating driver for mesh coding and decoding.
# class MeshCoderDriver
class MeshCoderDriver:
    """
    Supplies the symbol streams for the mesh decoder.
    Streams come directly from TopologicallyCompressedRepData fields.
    """

    def __init__(self, rep: TopologicallyCompressedRepData):
        self.rep = rep
        self._val_idx = 0
        self._grp_idx = 0
        self._flag_idx = 0
        # Degree symbols are stored per-context (8 contexts)
        self._deg_idx = [0] * 8
        self._use_flat_deg = False
        self._split_face_idx = 0
        self._split_pos_idx = 0

        self._val_stream = list(rep.vertex_valences or [])
        self._grp_stream = list(rep.vertex_groups or [])
        self._flag_stream = list(rep.vertex_flags or [])
        # Normalize face degree streams to 8 context lists (spec stores 8 streams)
        if rep.face_degrees is None:
            self._deg_streams = [[] for _ in range(8)]
        else:
            self._deg_streams = []
            for ctx in range(8):
                seq = rep.face_degrees[ctx] if ctx < len(rep.face_degrees) else []
                self._deg_streams.append(list(seq or []))
            if any(len(s) == 0 for s in self._deg_streams):
                # Some contexts are empty; prefer a single flat stream to avoid stalls
                self._use_flat_deg = True
        self._empty_deg_contexts = [i for i, s in enumerate(self._deg_streams) if len(s) == 0]
        self._last_deg_context = None
        self._last_deg_symbol = None
        self._last_deg_source = "unset"
        # Also keep a flat fallback stream if some contexts are empty
        self._deg_flat_stream = self._flatten(rep.face_degrees)
        self._deg_flat_idx = 0

        self._split_face_stream = list(rep.split_face_syms or [])
        self._split_pos_stream = list(rep.split_face_positions or [])

        self._attr_ctx_streams = {}
        fam = rep.face_attribute_masks or []
        for ctx, seq in enumerate(fam):
            self._attr_ctx_streams[ctx] = list(seq or [])
        self._attr_ctx_idx = {ctx: 0 for ctx in self._attr_ctx_streams}

        self._attr_large_stream = list(rep.high_degree_face_attribute_mask or [])
        self._attr_large_idx = 0

    @staticmethod
    def _flatten(obj) -> list:
        if obj is None:
            return []
        flat = []
        for seq in obj:
            if seq is None:
                continue
            if isinstance(seq, (list, tuple)):
                flat.extend(seq)
            else:
                flat.append(seq)
        return flat

    def _nextValSymbol(self) -> int:
        if self._val_idx >= len(self._val_stream):
            return -1
        v = self._val_stream[self._val_idx]
        self._val_idx += 1
        return int(v)

    def _nextFGrpSymbol(self) -> int:
        if self._grp_idx >= len(self._grp_stream):
            return 0
        v = self._grp_stream[self._grp_idx]
        self._grp_idx += 1
        return int(v)

    def _nextVtxFlagSymbol(self) -> int:
        if self._flag_idx >= len(self._flag_stream):
            return 0
        v = self._flag_stream[self._flag_idx]
        self._flag_idx += 1
        return int(v)

    def _faceCntxt(self, iVtx: int, vfm: DualVFMesh) -> int:
        cVal = vfm.valence(iVtx)
        print(iVtx)
        print(cVal)
        nKnownFaces = 0
        cKnownTotDeg = 0
        j = 0
        for i in range(cVal):
            j=j+1
            iTmpFace = vfm.face(iVtx, i)
            if not vfm.isValidFace(iTmpFace):
                continue
            nKnownFaces += 1
            cKnownTotDeg += vfm.degree(iTmpFace)
        iCCntxt = 0
        if cVal == 3:
            iCCntxt = 0 if cKnownTotDeg < nKnownFaces * 6 else (1 if cKnownTotDeg == nKnownFaces * 6 else 2)
        elif cVal == 4:
            iCCntxt = 3 if cKnownTotDeg < nKnownFaces * 4 else (4 if cKnownTotDeg == nKnownFaces * 4 else 5)
        elif cVal == 5:
            iCCntxt = 6
        else:
            iCCntxt = 7
        if iCCntxt == 2:
            print(
                f"_faceCntxt==2: vtx={iVtx} valence={cVal} "
                f"nKnownFaces={nKnownFaces} cKnownTotDeg={cKnownTotDeg}"
            )
        return iCCntxt

    def _nextDegSymbol(self, _context: int = 0) -> int:
        # If any context is empty, just drive from the flat stream to avoid stopping.
        if self._use_flat_deg:
            if self._deg_flat_idx < len(self._deg_flat_stream):
                v = self._deg_flat_stream[self._deg_flat_idx]
                self._deg_flat_idx += 1
                self._last_deg_context = _context
                self._last_deg_symbol = int(v)
                self._last_deg_source = "flat"
                return int(v)
            self._last_deg_context = _context
            self._last_deg_symbol = 0
            self._last_deg_source = "flat-empty"
            return 0

        if _context < 0 or _context >= len(self._deg_streams):
            self._last_deg_context = _context
            self._last_deg_symbol = 0
            self._last_deg_source = "invalid-context"
            return 0
        stream = self._deg_streams[_context]
        idx = self._deg_idx[_context]
        if idx < len(stream):
            self._deg_idx[_context] = idx + 1
            self._last_deg_context = _context
            self._last_deg_symbol = int(stream[idx])
            self._last_deg_source = "context"
            print(f"deg ctx={_context} idx={idx} val={stream[idx]}")
            return int(stream[idx])
        # Fallback to flat stream when context stream is exhausted
        if self._deg_flat_idx < len(self._deg_flat_stream):
            v = self._deg_flat_stream[self._deg_flat_idx]
            self._deg_flat_idx += 1
            self._last_deg_context = _context
            self._last_deg_symbol = int(v)
            self._last_deg_source = "flat-fallback"
            return int(v)
        self._last_deg_context = _context
        self._last_deg_symbol = 0
        self._last_deg_source = "exhausted"
        print(f"deg ctx={_context} idx={idx} val={stream[idx]}")
        exit()
        return 0

    def deg_debug(self) -> dict:
        return {
            "empty_contexts": list(self._empty_deg_contexts),
            "last_context": self._last_deg_context,
            "last_symbol": self._last_deg_symbol,
            "last_source": self._last_deg_source,
            "deg_idx": list(self._deg_idx),
            "deg_ctx_lens": [len(s) for s in self._deg_streams],
            "deg_flat_idx": self._deg_flat_idx,
            "deg_flat_len": len(self._deg_flat_stream),
        }

    def _nextAttrMaskSymbol(self, ctx: int):
        stream = self._attr_ctx_streams.get(ctx, [])
        idx = self._attr_ctx_idx.get(ctx, 0)
        if idx >= len(stream):
            return 0
        self._attr_ctx_idx[ctx] = idx + 1
        return int(stream[idx])

    def _nextAttrMaskSymbol_large(self) -> list[bool]:
        if self._attr_large_idx >= len(self._attr_large_stream):
            return []
        mask_val = int(self._attr_large_stream[self._attr_large_idx])
        self._attr_large_idx += 1
        bits = []
        for bit in range(64):
            bits.append(bool((mask_val >> bit) & 1))
        return bits

    def _nextSplitFaceSymbol(self) -> int:
        if self._split_face_idx >= len(self._split_face_stream):
            return -1
        v = self._split_face_stream[self._split_face_idx]
        self._split_face_idx += 1
        return int(v)

    def _nextSplitPosSymbol(self) -> int:
        if self._split_pos_idx >= len(self._split_pos_stream):
            return -1
        v = self._split_pos_stream[self._split_pos_idx]
        self._split_pos_idx += 1
        return int(v)
# This class serves as the abstract base class from which two concrete classes
# are derived to implement the core operations for a polygonal
# mesh coder or decoder. An instance of this object is used by the
# MeshCoderDriver to encode and decode polygonal meshes.
#
#  This class makes extensive use of DualVFMesh objects as the primary source and
#  destination mesh topology storage data structures. This mediating data
#  structure is necessary because the mesh coding scheme is deeply cooperative
#  with and dependent upon such a vertex-facet data structure. Please refer to
#  DualVFMesh for more information
class _MeshCodec:
    """Decode-side implementation of MeshCodec from the spec."""

    def __init__(self, driver: MeshCoderDriver):
        self._pTMC = driver
        self._pDstVFM: Optional[DualVFMesh] = None
        self._viActiveFaces: List[int] = []
        self._removedActiveFaces: set[int] = set()
        self._iFaceAttrCtr = 0

    def run(self) -> DualVFMesh:
        if self._pDstVFM is None:
            self._pDstVFM = DualVFMesh()
        self._pDstVFM.clear()
        self.clear()
        while True:
            if not self.runComponent():
                break
        return self._pDstVFM

    def clear(self):
        self._viActiveFaces.clear()
        self._removedActiveFaces.clear()
        self._iFaceAttrCtr = 0

    def runComponent(self):
        obFoundComponent = True
        obFoundComponent = self.initNewComponent()
        if not obFoundComponent:
            return False
        iFace = self.nextActiveFace()
        while iFace != -1:
            self.completeF(iFace)
            self.removeActiveFace(iFace)
            iFace = self.nextActiveFace()
        return True

    def initNewComponent(self):
        iVtx = self.ioVtxInit()
        if iVtx == -1:
            return False
        cVal = self._pDstVFM.valence(iVtx)
        for i in range(cVal):
            iFace = self.activateF(iVtx, i)
            if iFace == -2:
                raise RuntimeError("Mesh traversal failed")
        return True

    def completeF(self, iFace: int):
        jVtxSlot = self._pDstVFM.findVtxSlot(iFace, -1)
        iVSlot = 0
        while jVtxSlot != -1:
            iVtx = self.activateV(iFace, jVtxSlot)
            if iVtx < 0:
                logger.warning(
                    "Missing vertex symbol; skipping face completion: face=%d slot=%d",
                    iFace,
                    jVtxSlot,
                )
                if self._pDstVFM.numVts() > 0:
                    # Fill the slot with a fallback vertex to avoid infinite loops.
                    self._pDstVFM.setFaceVtx(iFace, jVtxSlot, 0)
                    jVtxSlot = self._pDstVFM.findVtxSlot(iFace, -1)
                    continue
                break
            if not (self._pDstVFM.vtx(iFace, jVtxSlot) == iVtx and self._pDstVFM.face(iVtx, iVSlot) == iFace):
                raise RuntimeError("FV consistency failed")
            self.completeV(iVtx, jVtxSlot)
            jVtxSlot = self._pDstVFM.findVtxSlot(iFace, -1)

    def activateF(self, iVtx: int, iVSlot: int) -> int:
        if iVtx < 0:
            return -1
        iFace = self.ioFace(iVtx, iVSlot)
        if iFace >= 0:
            if (
                not self._pDstVFM.setVtxFace(iVtx, iVSlot, iFace)
                or not self._pDstVFM.setFaceVtx(iFace, 0, iVtx)
            ):
                logger.error(
                    "activateF setVtxFace/setFaceVtx failed: vtx=%d vslot=%d face=%d",
                    iVtx,
                    iVSlot,
                    iFace,
                )
                return -2
            self.addActiveFace(iFace)
        elif iFace == -1:
            # Skip reuse/split faces; only keep newly created faces.
            return -1
        return iFace

    def activateV(self, iFace: int, iVSlot: int) -> int:
        iVtx = self.ioVtx(iFace, iVSlot)
        if iVtx < 0:
            return -1
        self._pDstVFM.setVtxFace(iVtx, 0, iFace)
        self.addVtxToFace(iVtx, 0, iFace, iVSlot)
        return iVtx

    def completeV(self, iVtx: int, iVSlot: int):
        vfm = self._pDstVFM
        cVal = vfm.valence(iVtx)

        # CCW
        vp = vfm.face(iVtx, 0)
        jp = iVSlot
        i = 1
        while True:
            vn = vfm.face(iVtx, i)
            if vn == -1:
                break
            jp = _dec_mod(jp, vfm.degree(vp))
            iVtx2 = vfm.vtx(vp, jp)
            if iVtx2 == -1:
                break
            jn = vfm.findVtxSlot(vn, iVtx2)
            if jn == -1:
                break
            jn = _dec_mod(jn, vfm.degree(vn))
            self.addVtxToFace(iVtx, i, vn, jn)
            vp = vn
            jp = jn
            i += 1
            if i >= cVal:
                return

        ilast = i
        vp = vfm.face(iVtx, 0)
        jp = iVSlot
        i = vfm.valence(iVtx) - 1
        while True:
            vn = vfm.face(iVtx, i)
            if vn == -1:
                break
            jp = _inc_mod(jp, vfm.degree(vp))
            iVtx2 = vfm.vtx(vp, jp)
            if iVtx2 == -1:
                break
            jn = vfm.findVtxSlot(vn, iVtx2)
            if jn == -1:
                break
            jn = _inc_mod(jn, vfm.degree(vn))
            self.addVtxToFace(iVtx, i, vn, jn)
            vp = vn
            jp = jn
            i -= 1
            if i < ilast:
                return

        for k in range(ilast, i + 1):
            iFace = self.activateF(iVtx, k)
            if iFace < -1:
                debug = self._pTMC.deg_debug()
                logger.error(
                    "activateF failed in completeV: vtx=%d vslot=%d k=%d valence=%d deg_debug=%s",
                    iVtx,
                    iVSlot,
                    k,
                    cVal,
                    debug,
                )
                print(f"Degree stream debug: {debug}")

    def addVtxToFace(self, iVtx: int, jFSlot: int, iFace: int, iVSlot: int):
        vfm = self._pDstVFM
        iVSlotCW = iVSlot
        iVSlotCCW = iVSlot
        iVSlotCCW = _inc_mod(iVSlotCCW, vfm.degree(iFace))
        iVSlotCW = _dec_mod(iVSlotCW, vfm.degree(iFace))

        vfm.setFaceVtx(iFace, iVSlot, iVtx)

        fp = vfm.vtx(iFace, iVSlotCW)
        if fp != -1:
            ip = vfm.findFaceSlot(fp, iFace)
            iVSlotCCW_loc = jFSlot
            iVSlotCCW_loc = _inc_mod(iVSlotCCW_loc, vfm.valence(iVtx))
            if vfm.face(iVtx, iVSlotCCW_loc) == -1:
                ip = _dec_mod(ip, vfm.valence(fp))
                vfm.setVtxFace(iVtx, iVSlotCCW_loc, vfm.face(fp, ip))

        fn = vfm.vtx(iFace, iVSlotCCW)
        if fn != -1:
            inn = vfm.findFaceSlot(fn, iFace)
            iVSlotCW_loc = jFSlot
            iVSlotCW_loc = _dec_mod(iVSlotCW_loc, vfm.valence(iVtx))
            if vfm.face(iVtx, iVSlotCW_loc) == -1:
                inn = _inc_mod(inn, vfm.valence(fn))
                vfm.setVtxFace(iVtx, iVSlotCW_loc, vfm.face(fn, inn))

    def addActiveFace(self, iFace: int):
        self._viActiveFaces.append(iFace)

    def nextActiveFace(self) -> int:
        while self._viActiveFaces and self._viActiveFaces[-1] in self._removedActiveFaces:
            self._viActiveFaces.pop()

        iFace = -1
        cLowestEmptyDegree = 9999999
        width = 16
        vfm = self._pDstVFM
        for idx in range(len(self._viActiveFaces) - 1, max(-1, len(self._viActiveFaces) - width), -1):
            iFace0 = self._viActiveFaces[idx]
            if iFace0 in self._removedActiveFaces:
                self._viActiveFaces.pop(idx)
                continue
            cEmpty = vfm.emptyFaceSlots(iFace0)
            if cEmpty < cLowestEmptyDegree:
                cLowestEmptyDegree = cEmpty
                iFace = iFace0
        return iFace

    def removeActiveFace(self, iFace: int):
        self._removedActiveFaces.add(iFace)

    def activeFaceOffset(self, iFace: int) -> int:
        cLen = len(self._viActiveFaces)
        for idx in range(cLen - 1, -1, -1):
            if self._viActiveFaces[idx] == iFace:
                return cLen - idx
        return -1

    def ioVtxInit(self) -> int:
        return self.ioVtx(-1, -1)

    def ioVtx(self, _iFace: int, _iVSlot: int) -> int:
        eSym = self._pTMC._nextValSymbol()
        iVtx = -1
        if eSym > -1:
            iVtx = self._pDstVFM.numVts()
            self._pDstVFM.newVtx(iVtx, eSym)
            self._pDstVFM.setVtxGrp(iVtx, self._pTMC._nextFGrpSymbol())
            self._pDstVFM.setVtxFlags(iVtx, self._pTMC._nextVtxFlagSymbol())
        return iVtx

    def ioFace(self, iVtx: int, _jFSlot: int) -> int:
        if iVtx < 0:
            return -1
        iCntxt = self._pTMC._faceCntxt(iVtx, self._pDstVFM)
        eSym = self._pTMC._nextDegSymbol(iCntxt)
        print(f"ioFace: vtx={iVtx} ctx={iCntxt} deg_symbol={eSym}")
        if eSym == 0:
            # Skip split/reuse faces; only create new faces.
            return -1

        iFace = self._pDstVFM.numFaces()
        cDeg = eSym
        print(f"ioFace: create new face={iFace} degree={cDeg}")
        nFaceAttrs = 0
        if cDeg <= DualVFMesh.cMBits:
            uAttrMask = self._pTMC._nextAttrMaskSymbol(
                max(0, min(7, cDeg - 2))
            )
            mask = int(uAttrMask)
            uMask = mask
            while uMask:
                nFaceAttrs += uMask & 1
                uMask >>= 1
            self._pDstVFM.newFace_smallMask(iFace, cDeg, nFaceAttrs, mask, 0)
        else:
            vbAttrMask = self._pTMC._nextAttrMaskSymbol_large()
            for bit in vbAttrMask:
                if bit:
                    nFaceAttrs += 1
            self._pDstVFM.newFace_bigMask(iFace, cDeg, nFaceAttrs, vbAttrMask, 0)

        if nFaceAttrs > cDeg:
            logger.warning(
                "Corrupt face attribute mask: %d attrs > degree %d; clamping",
                nFaceAttrs,
                cDeg,
            )
            nFaceAttrs = min(nFaceAttrs, cDeg)

        for iAttrSlot in range(nFaceAttrs):
            self._pDstVFM.setFaceAttr(iFace, iAttrSlot, self._iFaceAttrCtr)
            self._iFaceAttrCtr += 1
        return iFace

    def ioSplitFace(self, _iVtx: int, _jFSlot: int) -> int:
        eSym = self._pTMC._nextSplitFaceSymbol()
        if eSym < 0:
            return eSym
        iOffset = eSym
        cLen = len(self._viActiveFaces)
        if iOffset <= 0 or iOffset > cLen:
            logger.warning("Corrupt split face offset %d (len=%d); skipping", iOffset, cLen)
            return -1
        return self._viActiveFaces[cLen - iOffset]

    def ioSplitPos(self, _iVtx: int, _jFSlot: int) -> int:
        eSym = self._pTMC._nextSplitPosSymbol()
        return eSym
class DecodedMesh:
    """Container for decoded topology."""

    def __init__(
        self,
        face_vertices: list[list[int]],
        vertex_count: int,
        face_attr_indices: list[list[int]] = None,
    ):
        self.face_vertices = face_vertices
        self.vertex_count = vertex_count
        self.face_attr_indices = face_attr_indices or []
# This class implements the five abstract methods from
# MeshCodec to realize a mesh decoder.
class MeshDecoder:
    """Decoder facade exposing decode() -> DecodedMesh."""

    def __init__(self, rep_data: TopologicallyCompressedRepData):
        self.rep_data = rep_data
        self.driver = MeshCoderDriver(rep_data)

    def decode(self) -> DecodedMesh:
        codec = _MeshCodec(self.driver)
        vfm = codec.run()
        face_vertices: List[List[int]] = []
        face_attr_indices: List[List[int]] = []
        for i in range(vfm.numFaces()):
            deg = vfm.degree(i)
            verts = [vfm.vtx(i, slot) for slot in range(deg)]
            face_vertices.append(verts)
            attr_mask = vfm.attrMask(i)
            if attr_mask is None or deg == 0:
                face_attr_indices.append([-1] * deg)
                continue
            if isinstance(attr_mask, int):
                bits = [bool((attr_mask >> b) & 1) for b in range(deg)]
            else:
                bits = [bool(b) for b in attr_mask[:deg]]
            num_attrs = vfm.numAttrsOfFace(i)
            attrs = [vfm.faceAttr(i, s) for s in range(num_attrs)]
            if not attrs:
                face_attr_indices.append([-1] * deg)
                continue
            corner_attrs = [None] * deg
            corner_attrs[0] = attrs[0]
            attr_pos = 1
            # Apply the bit vector in the opposite traversal direction.
            for offset in range(1, deg):
                j = deg - offset
                if bits[j] and attr_pos < len(attrs):
                    corner_attrs[j] = attrs[attr_pos]
                    attr_pos += 1
                else:
                    corner_attrs[j] = corner_attrs[(j + 1) % deg]
            face_attr_indices.append(corner_attrs)

        vertex_records = self.rep_data.topologically_compressed_vertex_records
        vertex_count = getattr(
            vertex_records, "number_of_topological_vertices", vfm.numVts()
        ) or vfm.numVts()

        logger.info(
            "Decoded mesh (topology coder): %d faces, %d vertices",
            len(face_vertices),
            vertex_count,
        )
        return DecodedMesh(face_vertices, vertex_count, face_attr_indices)
