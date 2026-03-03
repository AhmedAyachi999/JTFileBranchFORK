import logging
import struct
import sys
from dataclasses import dataclass
from typing import Optional, List

from codec.i32Cdp2 import I32CDP2, PredictorType
from shape.topologicallyCompressedVertexRecords import TopologicallyCompressedVertexRecords
from util import byteStream as bs
from util.jt_hash import jt_hash16, jt_hash32_ints
from lsg.types import JtVersion
from shape.dual_vf_mesh import DualVFMesh

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
        logger.debug("creating from bytes")
        logger.debug((e_bytes.bytes[e_bytes.offset : e_bytes.offset + 30]).hex(" "))

        face_degrees = []
        for _ in range(8):
            face_degrees.append(I32CDP2.read_vec_i_32(e_bytes))

        vertex_valences = I32CDP2.read_vec_i_32(e_bytes)

        vertex_groups = I32CDP2.read_vec_i_32(e_bytes)
        vertex_flags = I32CDP2.read_vec_i_32(e_bytes, PredictorType.PredLag1)

        face_attribute_masks = []
        for _ in range(8):
            face_attribute_masks.append(I32CDP2.read_vec_i_32(e_bytes))
        face_attribute_masks8_30 = I32CDP2.read_vec_i_32(e_bytes)
        face_attribute_masks8_4 = I32CDP2.read_vec_i_32(e_bytes)
        high_degree_face_attribute_mask = bs.read_vec_i_32(e_bytes)
        split_face_syms = I32CDP2.read_vec_i_32(e_bytes, PredictorType.PredLag1)
        split_face_positions = I32CDP2.read_vec_i_32(e_bytes)
        read_hash = struct.unpack("<I", e_bytes.read(4))[0]
        topologically_compressed_vertex_records = TopologicallyCompressedVertexRecords.from_bytes(e_bytes)

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
    def compute_hash(
        cls,
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
    ):
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
        if len(face_attribute_masks) > 7:
            comp_hash = jt_hash32_ints(face_attribute_masks[7] or [], comp_hash)
        comp_hash = jt_hash32_ints(face_attribute_masks8_30, comp_hash)
        comp_hash = jt_hash32_ints(face_attribute_masks8_4, comp_hash)

        comp_hash = jt_hash32_ints(high_degree_face_attribute_mask, comp_hash)
        comp_hash = jt_hash32_ints(split_face_syms, comp_hash)
        comp_hash = jt_hash32_ints(split_face_positions, comp_hash)
        return comp_hash


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
        self._deg_idx = [0] * 8
        self._split_face_idx = 0
        self._split_pos_idx = 0

        self._val_stream = list(rep.vertex_valences or [])
        self._grp_stream = list(rep.vertex_groups or [])
        self._flag_stream = list(rep.vertex_flags or [])
        if rep.face_degrees is None:
            self._deg_streams = [[] for _ in range(8)]
        else:
            self._deg_streams = [list(seq or []) for seq in rep.face_degrees]
            if len(self._deg_streams) < 8:
                self._deg_streams.extend([[] for _ in range(8 - len(self._deg_streams))])
            elif len(self._deg_streams) > 8:
                self._deg_streams = self._deg_streams[:8]

        self._empty_deg_contexts = [i for i, s in enumerate(self._deg_streams) if len(s) == 0]
        self._last_deg_context = None
        self._last_deg_symbol = None
        self._last_deg_source = "unset"

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
        nKnownFaces = 0
        cKnownTotDeg = 0
        for i in range(cVal):
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
        print(f"[faceCntxt] vtx={iVtx} cVal={cVal} nKnownFaces={nKnownFaces} cKnownTotDeg={cKnownTotDeg} ctx={iCCntxt}")
        return iCCntxt

    def _nextDegSymbol(self, _context: int = 0) -> int:
        if _context < 0 or _context >= len(self._deg_streams):
            self._last_deg_context = _context
            self._last_deg_symbol = -1
            self._last_deg_source = "invalid-context"
            return -1

        stream = self._deg_streams[_context]
        idx = self._deg_idx[_context]

        if idx < len(stream):
            sym = int(stream[idx])
            self._deg_idx[_context] = idx + 1  # advance
            self._last_deg_context = _context
            self._last_deg_symbol = sym
            self._last_deg_source = "context"
            return sym

        self._last_deg_context = _context
        self._last_deg_symbol = -1
        self._last_deg_source = "exhausted"
        return -1

    def deg_debug(self) -> dict:
        return {
            "empty_contexts": list(self._empty_deg_contexts),
            "last_context": self._last_deg_context,
            "last_symbol": self._last_deg_symbol,
            "last_source": self._last_deg_source,
            "deg_idx": list(self._deg_idx),
            "deg_ctx_lens": [len(s) for s in self._deg_streams],
        }

    def _nextAttrMaskSymbol(self, ctx: int):
        stream = self._attr_ctx_streams.get(ctx, [])
        idx = self._attr_ctx_idx.get(ctx, 0)
        val = int(stream[idx]) if idx < len(stream) else 0
        if ctx == 7:
            if idx < len(self.rep.face_attribute_masks8_30 or []):
                val |= int(self.rep.face_attribute_masks8_30[idx]) << 30
            if idx < len(self.rep.face_attribute_masks8_4 or []):
                val |= int(self.rep.face_attribute_masks8_4[idx]) << 30
        self._attr_ctx_idx[ctx] = idx + 1
        return val

    def _nextAttrMaskSymbol_large(self, degree: int) -> list[bool]:
        if degree <= 0:
            return []
        n_words = (degree + 31) // 32
        if self._attr_large_idx >= len(self._attr_large_stream):
            return [False] * degree
        words = []
        for _ in range(n_words):
            if self._attr_large_idx < len(self._attr_large_stream):
                words.append(int(self._attr_large_stream[self._attr_large_idx]))
            else:
                words.append(0)
            self._attr_large_idx += 1
        bits: list[bool] = []
        for i in range(degree):
            word = words[i // 32]
            bit = (word >> (i % 32)) & 1
            bits.append(bool(bit))
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
class _MeshCodec:
    def __init__(self, driver: MeshCoderDriver):
        self._pTMC = driver
        self._pDstVFM: Optional[DualVFMesh] = None

        # Active face list (stack-like; newest at end)
        self._viActiveFaces: List[int] = []
        self._removedActiveFaces: set[int] = set()

        self._iFaceAttrCtr = 0
        self._current_component_id = -1
        self._face_component: dict[int, int] = {}
        self._vtx_component: dict[int, int] = {}

    def run(self, max_components: Optional[int] = None) -> DualVFMesh:
        if self._pDstVFM is None:
            self._pDstVFM = DualVFMesh()
        self._pDstVFM.clear()
        self.clear()

        components_run = 0
        while True:
            if not self.runComponent():
                break
            components_run += 1
            if max_components is not None and components_run >= max_components:
                break
        return self._pDstVFM

    def clear(self):
        self._viActiveFaces.clear()
        self._removedActiveFaces.clear()
        self._iFaceAttrCtr = 0
        self._current_component_id = -1
        self._face_component.clear()
        self._vtx_component.clear()

    def runComponent(self):
        obFoundComponent = self.initNewComponent()
        if not obFoundComponent:
            return False

        iFace = self.nextActiveFace()
        while iFace != -1:
            self.completeV(iFace)
            self.removeActiveFace(iFace)
            iFace = self.nextActiveFace()
        return True

    def initNewComponent(self):
        iVtx = self.ioVtxInit()
        if iVtx == -1:
            return False
        self._current_component_id += 1
        cVal = self._pDstVFM.valence(iVtx)
        for i in range(cVal):
            iFace = self.activateF(iVtx, i)
            if iFace == -2:
                raise RuntimeError("Mesh traversal failed")
        return True
    def addActiveFace(self, iFace: int):
        if iFace < 0:
            return
        if iFace in self._removedActiveFaces:
            return
        if iFace not in self._viActiveFaces:
            # enqueue at end (newest)
            self._viActiveFaces.append(iFace)

    def nextActiveFace(self) -> int:
        vfm = self._pDstVFM
        if vfm is None:
            return -1
        # Scan last 16 faces (newest) for lowest empty degree
        while self._viActiveFaces and self._viActiveFaces[-1] in self._removedActiveFaces:
            popped_face = self._viActiveFaces.pop()
            f = self._pDstVFM._vFaceEnts[popped_face]

        lowest_empty = 1_000_000_000
        chosen = -1
        width = 16
        start = max(0, len(self._viActiveFaces) - width)

        i = len(self._viActiveFaces) - 1
        while i >= start:
            f = self._viActiveFaces[i]
            if f in self._removedActiveFaces:
                del self._viActiveFaces[i]
                i -= 1
                continue

            empty_deg = vfm.emptyFaceSlots(f)
            if empty_deg < lowest_empty:
                lowest_empty = empty_deg
                chosen = f

            i -= 1

        return chosen

    def removeActiveFace(self, iFace: int):
        self._removedActiveFaces.add(iFace)

    def completeF(self, iVtx: int, iVSlot: int):
        vfm = self._pDstVFM
        cVal = vfm.valence(iVtx)

        # Walk CCW from face slot 0, attempting to link in as many
        # already-reachable faces as possible until we reach one
        # that is inactive.
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

        # Walk CW from face slot 0, attempting to link in as many
        # already-reachable faces as possible until we reach one
        # that is inactive
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

        # Activate remaining faces
        for k in range(ilast, i + 1):
            iFace = self.activateF(iVtx, k)
            if iFace < -1:
                debug = self._pTMC.deg_debug()
                raise RuntimeError(
                    "Mesh traversal failed in completeF: "
                    f"vtx={iVtx} vslot={iVSlot} k={k} valence={cVal} deg_debug={debug}"
                )

    def activateF(self, iVtx: int, iVSlot: int) -> int:
        if iVtx < 0:
            return -1
        iFace = self.ioFace(iVtx, iVSlot)
        if iFace == -2:
            return -2
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

                # Output last 16 initialized faces
                logger.error("Last 16 initialized faces:")
                num_faces = self._pDstVFM.numFaces()
                start_face = max(0, num_faces - 16)

                for i in range(start_face, num_faces):
                    if self._pDstVFM.isValidFace(i):
                        f = self._pDstVFM._vFaceEnts[i]
                        logger.error(
                            "  Face %d: cDeg=%d, iFVI=%d, cEmptyDeg=%d, cFaceAttrs=%d",
                            i,
                            f.cDeg,
                            f.iFVI,
                            f.cEmptyDeg,
                            f.cFaceAttrs,
                        )
                    else:
                        logger.error("  Face %d: INVALID", i)
                return -2
            self.addActiveFace(iFace)
            self._debug_activated_face(iFace)
            return iFace
        # Reuse/split face path
        iSplitFace = self.ioSplitFace(iVtx, iVSlot)
        if iSplitFace < 0:
            return -1
        iSplitPos = self.ioSplitPos(iVtx, iVSlot)
        if iSplitPos < 0:
            return -1

        self._pDstVFM.setVtxFace(iVtx, iVSlot, iSplitFace)
        self.addVtxToFace(iVtx, iVSlot, iSplitFace, iSplitPos)
        self._debug_activated_face(iSplitFace)
        return iSplitFace

    def activateV(self, iFace: int, iVSlot: int) -> int:
        iVtx = self.ioVtx(iFace, iVSlot)
        if iVtx == -1:
            return -1
        self._pDstVFM.setVtxFace(iVtx, 0, iFace)
        self.addVtxToFace(iVtx, 0, iFace, iVSlot)
        return iVtx

    def _debug_activated_face(self, iFace: int) -> None:
        vfm = self._pDstVFM
        if vfm is None or iFace < 0 or not vfm.isValidFace(iFace):
            return
        deg = vfm.degree(iFace)
        slots = [vfm.vtx(iFace, s) for s in range(deg)]
        print(f"Activated face {iFace} slots: {slots}")
        print(f"Activated face {iFace} degree: {deg}")

    def completeV(self, iFace: int):
        """
        Completes the VFMesh face iFace by calling activateV() and
        completeF() for each as-yet inactive incident vertices in the face's
        degree ring.
        """
        vfm = self._pDstVFM

        # While there is an empty vertex slot on the face
        iVSlot = 0
        while True:
            # Find next empty vertex slot (-1) on this face
            jVtxSlot = vfm.findVtxSlot(iFace, -1)
            if jVtxSlot == -1:
                break

            # Create and return a vertex iVtx, attaching it to iFace at vertex slot jVtxSlot
            iVtx = self.activateV(iFace, jVtxSlot)

            # Assert FV consistency
            if not (vfm.vtx(iFace, jVtxSlot) == iVtx and vfm.face(iVtx, iVSlot) == iFace):
                raise RuntimeError(
                    f"FV consistency error: face {iFace} slot {jVtxSlot} -> vtx {iVtx}, "
                    f"vtx {iVtx} slot {iVSlot} -> face {vfm.face(iVtx, iVSlot)}"
                )

            # Process the faces of iVtx starting from face slot jVtxSlot
            # where iVtx is incident on iFace
            self.completeF(iVtx, jVtxSlot)

            # Invariant "VF": vertex(iVtx).face(iVSlot) == iFace &&
            # face(iFace).vtx(jVtxSlot) == iVtx
    def addVtxToFace(self, iVtx: int, jFSlot: int, iFace: int, iVSlot: int):
        vfm = self._pDstVFM

        iVSlotCCW = _inc_mod(iVSlot, vfm.degree(iFace))
        iVSlotCW = _dec_mod(iVSlot, vfm.degree(iFace))

        vfm.setFaceVtx(iFace, iVSlot, iVtx)

        fp = vfm.vtx(iFace, iVSlotCW)
        if fp != -1:
            ip = vfm.findFaceSlot(fp, iFace)
            iVSlotCCW_loc = _inc_mod(jFSlot, vfm.valence(iVtx))
            if vfm.face(iVtx, iVSlotCCW_loc) == -1:
                ip = _dec_mod(ip, vfm.valence(fp))
                vfm.setVtxFace(iVtx, iVSlotCCW_loc, vfm.face(fp, ip))

        fn = vfm.vtx(iFace, iVSlotCCW)
        if fn != -1:
            inn = vfm.findFaceSlot(fn, iFace)
            iVSlotCW_loc = _dec_mod(jFSlot, vfm.valence(iVtx))
            if vfm.face(iVtx, iVSlotCW_loc) == -1:
                inn = _inc_mod(inn, vfm.valence(fn))
                vfm.setVtxFace(iVtx, iVSlotCW_loc, vfm.face(fn, inn))

    # -------------------------
    # Active face offset mapping (from end)
    # -------------------------
    def activeFaceOffset(self, iFace: int) -> int:
        """
        Offset from the end: 1 = newest (back), len = oldest (front).
        """
        c_len = len(self._viActiveFaces)
        for i in range(c_len - 1, -1, -1):
            if self._viActiveFaces[i] == iFace:
                return c_len - i
        return -1

    def ioVtxInit(self) -> int:
        return self.ioVtx(-1, -1)

    def ioVtx(self, _iFace: int, _iVSlot: int) -> int:
        eSym = self._pTMC._nextValSymbol()
        iVtx = -1
        if eSym > -1:
            iVtx = self._pDstVFM.numVts()
            self._pDstVFM.newVtx(iVtx, eSym)
            self._vtx_component[iVtx] = self._current_component_id
            self._pDstVFM.setVtxGrp(iVtx, self._pTMC._nextFGrpSymbol())
            self._pDstVFM.setVtxFlags(iVtx, self._pTMC._nextVtxFlagSymbol())
        return iVtx

    def ioFace(self, iVtx: int, _jFSlot: int) -> int:
        if iVtx < 0:
            return -1
        iCntxt = self._pTMC._faceCntxt(iVtx, self._pDstVFM)
        eSym = self._pTMC._nextDegSymbol(iCntxt)
        if eSym < 0:
            return -2
        if eSym == 0:
            return -1
        iFace = self._pDstVFM.numFaces()
        cDeg = eSym
        nFaceAttrs = 0

        if cDeg <= DualVFMesh.cMBits:
            uAttrMask = self._pTMC._nextAttrMaskSymbol(max(0, min(7, cDeg - 2)))
            mask = int(uAttrMask)
            uMask = mask
            while uMask:
                nFaceAttrs += uMask & 1
                uMask >>= 1
            self._pDstVFM.newFace_smallMask(iFace, cDeg, nFaceAttrs, mask, 0)
        else:
            vbAttrMask = self._pTMC._nextAttrMaskSymbol_large(cDeg)
            for bit in vbAttrMask:
                if bit:
                    nFaceAttrs += 1
            self._pDstVFM.newFace_bigMask(iFace, cDeg, nFaceAttrs, vbAttrMask, 0)

        self._face_component[iFace] = self._current_component_id

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
        """
        Split offset is from the end of the active list:
          1 = newest (back), len = oldest (front).
        """
        eSym = self._pTMC._nextSplitFaceSymbol()
        if eSym < 0:
            return eSym

        iOffset = int(eSym)
        cLen = len(self._viActiveFaces)
        if iOffset <= 0 or iOffset > cLen:
            logger.warning("Corrupt split face offset %d (len=%d); skipping", iOffset, cLen)
            return -1

        return self._viActiveFaces[cLen - iOffset]

    def ioSplitPos(self, _iVtx: int, _jFSlot: int) -> int:
        return self._pTMC._nextSplitPosSymbol()

    def component_maps(self) -> tuple[dict[int, int], dict[int, int]]:
        return self._face_component, self._vtx_component


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


@dataclass
class DecodedMeshComponent:
    component_id: int
    face_vertices: list[list[int]]
    vertex_ids: list[int]


class MeshDecoder:
    """Decoder facade exposing decode() -> DecodedMesh."""
    def __init__(self, rep_data: TopologicallyCompressedRepData):
        self.rep_data = rep_data
        self.driver = MeshCoderDriver(rep_data)

    def decode(self, max_components: Optional[int] = None) -> DecodedMesh:
        codec = _MeshCodec(self.driver)
        vfm = codec.run(max_components=max_components)

        face_vertices: List[List[int]] = []
        face_attr_indices: List[List[int]] = []

        # The DualVFMesh stores the dual representation
        for iVtx in range(vfm.numVts()):
            if vfm.vtxGrp(iVtx) < 0:
                continue
            valence = vfm.valence(iVtx)

            # Get incident faces
            verts = [vfm.face(iVtx, slot) for slot in range(valence)]
            face_vertices.append(verts)
            corner_attrs = [-1] * valence
            face_attr_indices.append(corner_attrs)

        # The vertex count is the number of faces in the dual mesh
        vertex_count = vfm.numFaces()

        logger.info("Decoded mesh (topology coder): %d faces, %d vertices", len(face_vertices), vertex_count)
        return DecodedMesh(face_vertices, vertex_count, face_attr_indices)

    def decode_components(
        self,
        max_components: Optional[int] = None,
        remap_vertices: bool = False,
    ) -> list[DecodedMeshComponent]:
        codec = _MeshCodec(self.driver)
        vfm = codec.run(max_components=max_components)

        face_comp, _vtx_comp = codec.component_maps()
        if not face_comp:
            return []

        component_ids = sorted(set(face_comp.values()))
        components: list[DecodedMeshComponent] = []

        for cid in component_ids:
            face_indices = [i for i, c in face_comp.items() if c == cid]
            face_vertices = []
            vertex_ids_set = set()

            for iFace in face_indices:
                deg = vfm.degree(iFace)
                verts = [vfm.vtx(iFace, slot) for slot in range(deg)]
                face_vertices.append(verts)
                vertex_ids_set.update(verts)

            vertex_ids = sorted(v for v in vertex_ids_set if v is not None and v >= 0)

            if remap_vertices:
                remap = {v: idx for idx, v in enumerate(vertex_ids)}
                face_vertices = [[remap[v] for v in face if v in remap] for face in face_vertices]

            components.append(
                DecodedMeshComponent(
                    component_id=cid,
                    face_vertices=face_vertices,
                    vertex_ids=vertex_ids,
                )
            )

        return components
