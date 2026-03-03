
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

from shape.topologicallyCompressedRepData import MeshDecoder


@dataclass
class TriangleMeshData:
    vertices: np.ndarray  # (N, 3) float64
    triangles: np.ndarray  # (M, 3) int32


def _fan_triangulate(face: Iterable[int]) -> list[tuple[int, int, int]]:
    verts = list(face)
    if len(verts) < 3:
        return []
    v0 = verts[0]
    out: list[tuple[int, int, int]] = []
    for i in range(1, len(verts) - 1):
        v1 = verts[i]
        v2 = verts[i + 1]
        if v0 != v1 and v1 != v2 and v2 != v0:
            out.append((v0, v1, v2))
    return out


def build_triangle_mesh_from_lod(lod, max_components: Optional[int] = 1) -> TriangleMeshData:
    """
    Build a triangle mesh (vertices + triangle indices) from a Shape LOD.

    This mirrors the JT v9.5 topologically-compressed pipeline:
    - decode topology via MeshDecoder
    - map vertex records -> XYZ coordinates
    - fan-triangulate polygonal faces
    """
    rep_data = lod.vertex_shape_LOD_data.topo_mesh_compressed_lod_data.topo_mesh_compressed_rep_data
    coord_arr = rep_data.topologically_compressed_vertex_records.compressed_vertex_coordinate_array
    if coord_arr is None or coord_arr.vertex_coordinates is None:
        return TriangleMeshData(np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.int32))

    coords = coord_arr.vertex_coordinates  # expected (3, N)
    if len(coords) < 3:
        return TriangleMeshData(np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.int32))

    x, y, z = coords[:3]
    vertices = np.column_stack((x, y, z)).astype(np.float64, copy=False)
    n_verts = int(vertices.shape[0])
    if n_verts == 0:
        return TriangleMeshData(vertices, np.zeros((0, 3), dtype=np.int32))

    # Use component-based decode so faces are built from DualVFMesh faces.
    components = MeshDecoder(rep_data).decode_components(
        max_components=max_components,
        remap_vertices=False,
    )
    faces: list[list[int]] = []
    for comp in components:
        faces.extend(comp.face_vertices)

    triangles: list[tuple[int, int, int]] = []
    for face in faces:
        # validate local indices
        local = [v for v in face if 0 <= v < n_verts]
        triangles.extend(_fan_triangulate(local))

    tri_arr = np.asarray(triangles, dtype=np.int32) if triangles else np.zeros((0, 3), dtype=np.int32)
    return TriangleMeshData(vertices, tri_arr)
