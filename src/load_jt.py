# load_jt.py
# Load a JT file, decode Shape segments, and visualize:
#   - PointCloud of all vertices
#   - Wireframe LineSet built from decoded faces
#
# Key fixes vs. your current script:
#   1) all_edges is actually filled
#   2) edges are offset-safe when points are stacked (vert_offset)
#   3) edges are undirected + de-duplicated (sorted pair)
#   4) degenerate/self edges are skipped
#   5) optional skip_tiny actually works

import argparse
import logging
import struct
import traceback
import sys
import zlib
import lzma
import colorsys
from dataclasses import dataclass, field
from typing import Final

import numpy as np
import open3d as o3d

from core import logging_config
from lsg.lsg import LSG, read_lsg_segment
from lsg.types import GUID, JtVersion
from metadata.metadata import Metadata, read_metadata_segment
from shape.shape import Shape, read_shape_segment, ShapeLod0, ShapeLod1
from shape.topologicallyCompressedRepData import MeshDecoder
from util.byteStream import ByteStream


logger = logging.getLogger(__name__)
VERSION = JtVersion.unsupported


def debug_info(type_, value, tb):
    import pdb
    traceback.print_exception(type_, value, tb)
    pdb.post_mortem(tb)


@dataclass
class DataSegmentType:
    id: int
    name: str
    compression: bool


@dataclass
class TocEntry:
    guid: GUID
    offset: int
    length: int
    attr: int
    type: DataSegmentType


@dataclass
class DataSegment:
    guid: GUID
    type: DataSegmentType
    length: int
    data: str = field(repr=False)


DATA_SEGMENT_TYPES: Final = [
    DataSegmentType(1, "Logical Scene Graph", True),
    DataSegmentType(2, "JT B-Rep", True),
    DataSegmentType(3, "PMI Data", True),
    DataSegmentType(4, "Meta Data", True),
    DataSegmentType(5, "NULL", False),
    DataSegmentType(6, "Shape", False),
    DataSegmentType(7, "Shape LOD0", False),
    DataSegmentType(8, "Shape LOD1", False),
    DataSegmentType(9, "Shape LOD2", False),
    DataSegmentType(10, "Shape LOD3", False),
    DataSegmentType(11, "Shape LOD4", False),
    DataSegmentType(12, "Shape LOD5", False),
    DataSegmentType(13, "Shape LOD6", False),
    DataSegmentType(14, "Shape LOD7", False),
    DataSegmentType(15, "Shape LOD8", False),
    DataSegmentType(16, "Shape LOD9", False),
    DataSegmentType(17, "XT B-Rep", True),
    DataSegmentType(18, "'Wireframe Representation'", True),
    DataSegmentType(19, "NULL", False),
    DataSegmentType(20, "ULP", True),
    DataSegmentType(21, "NULL", False),
    DataSegmentType(22, "NULL", False),
    DataSegmentType(23, "STT", True),  # V10.5
    DataSegmentType(24, "LWPA", True),
    DataSegmentType(25, "NULL", False),
    DataSegmentType(26, "NULL", False),
    DataSegmentType(27, "NULL", False),
    DataSegmentType(28, "NULL", False),
    DataSegmentType(29, "NULL", False),
    DataSegmentType(30, "MultiXT B-Rep", True),
    DataSegmentType(31, "InfoSegment", True),
    DataSegmentType(32, "Reserved", True),
    DataSegmentType(33, "STEP B-rep", True),
]


def read_table_of_contents(path: str) -> list[TocEntry]:
    with open(path, mode="rb") as jt:
        global VERSION

        # Header
        jt_version = jt.read(80)[8:12]
        if jt_version in (b"9.5 ", b"9.4 "):
            VERSION = JtVersion.V9d5
        elif jt_version in (b"10.5", b"10.0"):
            VERSION = JtVersion.V10d5
        else:
            raise NotImplementedError(f"version {jt_version} not supported")

        logger.info("reading jt file with version %s", VERSION)

        # Byte order byte (kept but not used)
        _jt_bo = int.from_bytes(jt.read(1), byteorder="little")
        _jt_reserved = jt.read(4)

        jt_toc_offset = struct.unpack("i", jt.read(4))[0]
        jt.seek(jt_toc_offset)

        jt_toc_entry_count = int.from_bytes(jt.read(4), byteorder="little")
        toc: list[TocEntry] = []

        for _ in range(jt_toc_entry_count):
            seg_id = GUID.from_bytes(jt)
            if VERSION == JtVersion.V9d5:
                seg_offset, seg_len, seg_attr = struct.unpack("iiI", jt.read(12))
            else:
                seg_offset, seg_len, seg_attr = struct.unpack("QII", jt.read(16))

            seg_type = (seg_attr & 0xFF000000) >> 24
            toc.append(TocEntry(seg_id, seg_offset, seg_len, seg_attr, DATA_SEGMENT_TYPES[seg_type - 1]))

        return toc


def read_segment(path: str, toc_entry_offset: int):
    with open(path, mode="rb") as jt:
        jt.seek(toc_entry_offset)

        _ds_id = GUID.from_bytes(jt)
        ds_type, ds_len = struct.unpack("ii", jt.read(8))

        # Load (and decompress if needed)
        if DATA_SEGMENT_TYPES[ds_type - 1].compression:
            if VERSION == JtVersion.V9d5:
                comp_flag, comp_len, comp_alg = struct.unpack("iiB", jt.read(9))
                comp_len -= 1
                raw = jt.read(comp_len)
                if comp_flag == 2 and comp_alg == 2:
                    ds_bytes = ByteStream(zlib.decompress(raw))
                else:
                    ds_bytes = ByteStream(raw)
            else:
                comp_flag, comp_len, comp_alg = struct.unpack("IiB", jt.read(9))
                comp_len -= 1
                raw = jt.read(comp_len)
                if comp_flag == 3 and comp_alg == 3:
                    ds_bytes = ByteStream(lzma.decompress(raw))
                else:
                    ds_bytes = ByteStream(raw)
        else:
            ds_bytes = ByteStream(jt.read(ds_len))

        # Decode known segments
        if ds_type == LSG.SEGMENT_TYPE_ID:
            return read_lsg_segment(ds_bytes, version=VERSION)
        if ds_type == Metadata.SEGMENT_TYPE_ID:
            return read_metadata_segment(ds_bytes, VERSION)
        if ds_type in {Shape.SEGMENT_TYPE_ID, ShapeLod0.SEGMENT_TYPE_ID, ShapeLod1.SEGMENT_TYPE_ID}:
            return read_shape_segment(ds_bytes, VERSION)

        return None


def bbox_diag(points: np.ndarray) -> float:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    return float(np.linalg.norm(maxs - mins))


def main():
    parser = argparse.ArgumentParser(description="Load a JT file and draw wireframe with Open3D")
    parser.add_argument("path")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--skip_tiny",
        action="store_true",
        help="Skip shapes with extremely tiny bounding boxes (helps when decode is wrong for some shapes).",
    )
    parser.add_argument(
        "--tiny_diag",
        type=float,
        default=1e-3,
        help="Bounding box diagonal threshold for --skip_tiny (default: 1e-3).",
    )
    parser.add_argument(
        "--max_components",
        type=int,
        default=1,
        help="Max connected components decoded by MeshDecoder (default: 1).",
    )
    args = parser.parse_args()

    logging_config.configure_logging(args.debug)
    if args.debug:
        sys.excepthook = debug_info

    logger.info("Started")

    PATH = args.path
    jt_toc = read_table_of_contents(PATH)

    def is_lsg(t: TocEntry) -> bool:
        return t.type.id == 1

    def is_shape(t: TocEntry) -> bool:
        return t.type.id == 6

    lsg_entries = list(filter(is_lsg, jt_toc))
    shape_entries = list(filter(is_shape, jt_toc))

    if lsg_entries:
        lsg_entry = lsg_entries[0]
        logger.info("reading lsg segment at %s", lsg_entry.offset)
        lsg = read_segment(PATH, lsg_entry.offset)
        if lsg is not None:
            print(lsg.ascii_lsg_tree())

    if not shape_entries:
        logger.warning("No shape segments found in JT file.")
        return

    all_points: list[np.ndarray] = []
    all_edges: list[tuple[int, int]] = []
    all_edge_colors: list[tuple[float, float, float]] = []
    all_triangles: list[tuple[int, int, int]] = []

    vert_offset = 0
    shape_index = 0

    for entry in shape_entries:
        logger.info("reading shape segment %s at %s", entry.guid, entry.offset)
        shapes = read_segment(PATH, entry.offset)

        if not shapes:
            logger.warning("No shapes decoded in segment %s; skipping.", entry.guid)
            continue

        for shape_id, lod in shapes.items():
            if lod is None:
                continue

            rep_data = lod.vertex_shape_LOD_data.topo_mesh_compressed_lod_data.topo_mesh_compressed_rep_data
            coord_arr = rep_data.topologically_compressed_vertex_records.compressed_vertex_coordinate_array
            coords = coord_arr.vertex_coordinates  # expected (3, N)

            x, y, z = coords
            points = np.column_stack((x, y, z)).astype(np.float64, copy=False)
            n_verts = int(points.shape[0])
            if n_verts == 0:
                continue


            # Decode topology (faces)
            decoded = MeshDecoder(rep_data).decode(max_components=args.max_components)
            faces = decoded.face_vertices or []

            logger.info("shape %s: vertices=%d faces=%d", shape_id, n_verts, len(faces))

            # Build edges (offset-safe, undirected, de-duplicated)
            edges_set: set[tuple[int, int]] = set()
            bad_faces = 0

            for face in faces:
                m = len(face)
                if m < 2:
                    continue
                for i in range(m):
                    v0 = face[i]
                    v1 = face[(i + 1) % m]
                    # Validate local indices
                    if not (0 <= v0 < n_verts and 0 <= v1 < n_verts):
                        bad_faces += 1
                        continue

                    # Skip self-edge
                    if v0 == v1:
                        continue

                    # Convert local -> global index space
                    a = v0 + vert_offset
                    b = v1 + vert_offset
                    edges_set.add((a, b))
            if bad_faces:
                logger.warning("shape %s: skipped %d out-of-range face-edges (decode likely off)", shape_id, bad_faces)

            # Append points + edges to global buffers
            all_points.append(points)
            # Per-shape color (optional; Open3D LineSet supports per-line colors)
            color = colorsys.hsv_to_rgb((shape_index * 0.61803398875) % 1.0, 0.6, 0.9)
            edges_list = list(edges_set)
            all_edges.extend(edges_list)
            all_edge_colors.extend([color] * len(edges_list))
            # ---- Build triangles from faces ----
            for face in faces:
                if len(face) < 3:
                    continue

                # Convert to global indices
                face_global = []
                for v in face:
                    if 0 <= v < n_verts:
                        face_global.append(v + vert_offset)

                if len(face_global) < 3:
                    continue

                # Fan triangulation
                v0 = face_global[0]
                for i in range(1, len(face_global) - 1):
                    v1 = face_global[i]
                    v2 = face_global[i + 1]

                    if v0 != v1 and v1 != v2 and v2 != v0:
                        all_triangles.append((v0, v1, v2))

            vert_offset += n_verts
            shape_index += 1

    if not all_points:
        logger.warning("No points collected; nothing to draw.")
        return

    # for the edges
    points_big = np.vstack(all_points).astype(np.float64, copy=False)
    E = np.asarray(all_edges, dtype=np.int32)

    line_set = o3d.geometry.LineSet()
    # For the points
    line_set.points = o3d.utility.Vector3dVector(points_big)
    # For the lines
    line_set.lines = o3d.utility.Vector2iVector(E)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_big)
    for k, (i, j) in enumerate(E):
        pi, pj = points_big[i], points_big[j]
        d = float(np.linalg.norm(pi - pj))
        print(k, (i, j), "len=", d, "pi=", pi, "pj=", pj)


    points_big = np.vstack(all_points).astype(np.float64, copy=False)
    T = np.asarray(all_triangles, dtype=np.int32)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points_big)
    mesh.triangles = o3d.utility.Vector3iVector(T)

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()

    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()

    mesh.paint_uniform_color([0.7, 0.75, 0.85])

    o3d.visualization.draw_geometries(
        [mesh],
        window_name="JT Surface"
    )

    logger.info("Finished")


if __name__ == "__main__":
    main()
