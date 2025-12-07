from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Union


class DualVFMesh:
    # Number of optimized mask bits (same as C++ cMBits)
    cMBits = 64

    # ---------- Inner record types ----------

    @dataclass
    class VtxEnt:
        cVal: int = 0          # vertex valence
        uFlags: int = 0        # user flags (cover/primal etc.)
        iVGrp: int = -1        # vertex group id
        iVFI: int = -1         # index into _viVtxFaceIndices

    @dataclass
    class FaceEnt:
        cDeg: int = 0          # face degree
        cEmptyDeg: int = 0     # how many slots still -1
        cFaceAttrs: int = 0    # number of face attributes
        uFlags: int = 0        # user flags
        # For cDeg <= cMBits this is an int bit-mask.
        # For cDeg > cMBits this is a list[bool] (BitVec equivalent).
        attr_mask: Optional[Union[int, List[bool]]] = None
        iFVI: int = -1         # index into _viFaceVtxIndices
        iFAI: int = -1         # index into _viFaceAttrIndices

    # ---------- Constructor ----------

    def __init__(self) -> None:
        # One entry per vertex
        self._vVtxEnts: List[DualVFMesh.VtxEnt] = []
        # One entry per face
        self._vFaceEnts: List[DualVFMesh.FaceEnt] = []
        # For *all* vertices, concatenated:
        #   [faces of vtx0..., faces of vtx1..., ...]
        self._viVtxFaceIndices: List[int] = []
        # For *all* faces, concatenated:
        #   [vertices of face0..., vertices of face1..., ...]
        self._viFaceVtxIndices: List[int] = []
        # For *all* face attributes:
        self._viFaceAttrIndices: List[int] = []

    # Optional, used later by MeshCodec.run()
    def clear(self) -> None:
        self._vVtxEnts.clear()
        self._vFaceEnts.clear()
        self._viVtxFaceIndices.clear()
        self._viFaceVtxIndices.clear()
        self._viFaceAttrIndices.clear()

    # ---------- Small helpers to grow lists safely ----------

    def _ensure_vtx_index(self, iVtx: int) -> None:
        while len(self._vVtxEnts) <= iVtx:
            self._vVtxEnts.append(DualVFMesh.VtxEnt())

    def _ensure_face_index(self, iFace: int) -> None:
        while len(self._vFaceEnts) <= iFace:
            self._vFaceEnts.append(DualVFMesh.FaceEnt())

    # ---------- Queries (almost 1:1 with C++) ----------

    def numVts(self) -> int:
        return len(self._vVtxEnts)

    def numFaces(self) -> int:
        return len(self._vFaceEnts)

    def numAttrs(self) -> int:
        return len(self._viFaceAttrIndices)

    def numAttrsOfFace(self, iFace: int) -> int:
        return self._vFaceEnts[iFace].cFaceAttrs

    def valence(self, iVtx: int) -> int:
        return self._vVtxEnts[iVtx].cVal

    def degree(self, iFace: int) -> int:
        return self._vFaceEnts[iFace].cDeg

    def face(self, iVtx: int, iFaceSlot: int) -> int:
        v = self._vVtxEnts[iVtx]
        return self._viVtxFaceIndices[v.iVFI + iFaceSlot]

    def vtx(self, iFace: int, iVtxSlot: int) -> int:
        f = self._vFaceEnts[iFace]
        return self._viFaceVtxIndices[f.iFVI + iVtxSlot]

    def emptyFaceSlots(self, iFace: int) -> int:
        return self._vFaceEnts[iFace].cEmptyDeg

    # ---------- Vertex side ----------

    def isValidVtx(self, iVtx: int) -> bool:
        if 0 <= iVtx < len(self._vVtxEnts):
            return self._vVtxEnts[iVtx].cVal != 0
        return False

    def newVtx(self, iVtx: int, iValence: int, uFlags: int = 0) -> bool:
        """Create/initialize vertex iVtx with given valence and flags."""
        self._ensure_vtx_index(iVtx)
        v = self._vVtxEnts[iVtx]

        if v.cVal != iValence:
            v.cVal = iValence
            v.uFlags = uFlags
            v.iVFI = len(self._viVtxFaceIndices)

            # Allocate iValence slots, all -1 (no incident faces yet)
            self._viVtxFaceIndices.extend([-1] * iValence)

        return True

    def setVtxGrp(self, iVtx: int, iVGrp: int) -> bool:
        self._ensure_vtx_index(iVtx)
        self._vVtxEnts[iVtx].iVGrp = iVGrp
        return True

    def setVtxFlags(self, iVtx: int, uFlags: int) -> bool:
        self._ensure_vtx_index(iVtx)
        self._vVtxEnts[iVtx].uFlags = uFlags
        return True

    def vtxGrp(self, iVtx: int) -> int:
        if 0 <= iVtx < len(self._vVtxEnts):
            return self._vVtxEnts[iVtx].iVGrp
        return -1

    def vtxFlags(self, iVtx: int) -> int:
        if 0 <= iVtx < len(self._vVtxEnts):
            return self._vVtxEnts[iVtx].uFlags
        return 0

    # ---------- Face side ----------

    def isValidFace(self, iFace: int) -> bool:
        if 0 <= iFace < len(self._vFaceEnts):
            return self._vFaceEnts[iFace].cDeg != 0
        return False

    def newFace_smallMask(
        self,
        iFace: int,
        cDegree: int,
        cFaceAttrs: int = 0,
        uFaceAttrMask: int = 0,
        uFlags: int = 0,
    ) -> bool:
        """
        cDeg <= 64 case: attr mask is a 64-bit integer.
        Equivalent to C++ newFace(..., UInt64 uFaceAttrMask, ...).
        """
        self._ensure_face_index(iFace)
        f = self._vFaceEnts[iFace]

        if f.cDeg != cDegree:
            f.cDeg = cDegree
            f.cEmptyDeg = cDegree
            f.cFaceAttrs = cFaceAttrs
            f.uFlags = uFlags
            f.attr_mask = int(uFaceAttrMask)

            f.iFVI = len(self._viFaceVtxIndices)
            f.iFAI = len(self._viFaceAttrIndices)

            # Allocate slots for vertices and attrs
            self._viFaceVtxIndices.extend([-1] * cDegree)
            if cFaceAttrs > 0:
                self._viFaceAttrIndices.extend([-1] * cFaceAttrs)

        return True

    def newFace_bigMask(
        self,
        iFace: int,
        cDegree: int,
        cFaceAttrs: int,
        faceAttrMaskBits: List[bool],
        uFlags: int = 0,
    ) -> bool:
        """
        cDeg > 64 case: attr mask stored as list[bool] (BitVec).
        Equivalent to C++ newFace(..., const BitVec*).
        """
        self._ensure_face_index(iFace)
        f = self._vFaceEnts[iFace]

        if f.cDeg != cDegree:
            f.cDeg = cDegree
            f.cEmptyDeg = cDegree
            f.cFaceAttrs = cFaceAttrs
            f.uFlags = uFlags
            f.attr_mask = list(faceAttrMaskBits)

            f.iFVI = len(self._viFaceVtxIndices)
            f.iFAI = len(self._viFaceAttrIndices)

            self._viFaceVtxIndices.extend([-1] * cDegree)
            if cFaceAttrs > 0:
                self._viFaceAttrIndices.extend([-1] * cFaceAttrs)

        return True

    def setFaceFlags(self, iFace: int, uFlags: int) -> bool:
        self._ensure_face_index(iFace)
        self._vFaceEnts[iFace].uFlags = uFlags
        return True

    def faceFlags(self, iFace: int) -> int:
        if 0 <= iFace < len(self._vFaceEnts):
            return self._vFaceEnts[iFace].uFlags
        return 0

    def setFaceAttr(self, iFace: int, iAttrSlot: int, iFaceAttr: int) -> bool:
        f = self._vFaceEnts[iFace]
        base = f.iFAI
        self._viFaceAttrIndices[base + iAttrSlot] = iFaceAttr
        return True

    def faceAttr(self, iFace: int, iAttrSlot: int) -> int:
        if 0 <= iFace < len(self._vFaceEnts):
            f = self._vFaceEnts[iFace]
            if 0 <= iAttrSlot < f.cFaceAttrs:
                base = f.iFAI
                return self._viFaceAttrIndices[base + iAttrSlot]
        return 0

    def attrMask(self, iFace: int) -> Optional[Union[int, List[bool]]]:
        return self._vFaceEnts[iFace].attr_mask

    # ---------- Topology connection ----------

    def setVtxFace(self, iVtx: int, iFaceSlot: int, iFace: int) -> bool:
        v = self._vVtxEnts[iVtx]
        self._viVtxFaceIndices[v.iVFI + iFaceSlot] = iFace
        return True

    def setFaceVtx(self, iFace: int, iVtxSlot: int, iVtx: int) -> bool:
        f = self._vFaceEnts[iFace]
        idx = f.iFVI + iVtxSlot
        old = self._viFaceVtxIndices[idx]

        # Decrease emptyDeg if we are filling a previously empty / different slot
        if old != iVtx:
            f.cEmptyDeg -= 1

        self._viFaceVtxIndices[idx] = iVtx
        return True

    # ---------- Search helpers ----------

    def findVtxSlot(self, iFace: int, iTargVtx: int) -> int:
        """
        Search all vertices of face iFace for iTargVtx, return slot index or -1.
        """
        f = self._vFaceEnts[iFace]
        base = f.iFVI
        for slot in range(f.cDeg):
            if self._viFaceVtxIndices[base + slot] == iTargVtx:
                return slot
        return -1

    def findFaceSlot(self, iVtx: int, iTargFace: int) -> int:
        """
        Search all faces incident on vertex iVtx for iTargFace, return slot or -1.
        """
        v = self._vVtxEnts[iVtx]
        base = v.iVFI
        for slot in range(v.cVal):
            if self._viVtxFaceIndices[base + slot] == iTargFace:
                return slot
        return -1
