import argparse
import json
import logging
import lzma
import struct
import sys
import traceback
import zlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import numpy as np

from core import logging_config
from lsg.lsg import LSG, read_lsg_segment
from lsg.types import GUID, JtVersion
from metadata.metadata import Metadata, read_metadata_segment
from shape.shape import (
    Shape,
    ShapeLod0,
    ShapeLod1,
    ShapeLod2,
    ShapeLod3,
    ShapeLod4,
    ShapeLod5,
    ShapeLod6,
    ShapeLod7,
    ShapeLod8,
    ShapeLod9,
    read_shape_segment,
)
from shape.topologicallyCompressedRepData import MeshCoderDriver, _MeshCodec
from util.byteStream import ByteStream


def debug_info(exc_type, value, tb):
    import pdb

    traceback.print_exception(exc_type, value, tb)
    pdb.post_mortem(tb)


VERSION = JtVersion.unsupported
logger = logging.getLogger(__name__)


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
class DecodedMesh:
    name: str
    points: np.ndarray
    triangles: np.ndarray


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
    DataSegmentType(18, "Wireframe Representation", True),
    DataSegmentType(19, "NULL", False),
    DataSegmentType(20, "ULP", True),
    DataSegmentType(21, "NULL", False),
    DataSegmentType(22, "NULL", False),
    DataSegmentType(23, "STT", True),
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

SHAPE_SEGMENT_TYPE_IDS: Final = {
    Shape.SEGMENT_TYPE_ID,
    ShapeLod0.SEGMENT_TYPE_ID,
    ShapeLod1.SEGMENT_TYPE_ID,
    ShapeLod2.SEGMENT_TYPE_ID,
    ShapeLod3.SEGMENT_TYPE_ID,
    ShapeLod4.SEGMENT_TYPE_ID,
    ShapeLod5.SEGMENT_TYPE_ID,
    ShapeLod6.SEGMENT_TYPE_ID,
    ShapeLod7.SEGMENT_TYPE_ID,
    ShapeLod8.SEGMENT_TYPE_ID,
    ShapeLod9.SEGMENT_TYPE_ID,
}


def read_table_of_contents(path: Path) -> list[TocEntry]:
    with open(path, mode="rb") as jt:
        global VERSION
        jt_version = jt.read(80)[8:12]
        if jt_version in {b"9.5 ", b"9.4 "}:
            VERSION = JtVersion.V9d5
        elif jt_version in {b"10.5", b"10.0"}:
            VERSION = JtVersion.V10d5
        else:
            raise NotImplementedError(f"version {jt_version!r} not supported")

        logger.info("reading jt file with version %s", VERSION)
        jt.read(1)
        jt.read(4)
        jt_toc_offset = struct.unpack("i", jt.read(4))[0]

        jt.seek(jt_toc_offset)
        entry_count = int.from_bytes(jt.read(4), byteorder="little")
        toc_entries: list[TocEntry] = []
        for _ in range(entry_count):
            seg_id = GUID.from_bytes(jt)
            if VERSION == JtVersion.V9d5:
                seg_offset, seg_len, seg_attr = struct.unpack("iiI", jt.read(12))
            else:
                seg_offset, seg_len, seg_attr = struct.unpack("QII", jt.read(16))

            seg_type = (seg_attr & 0xFF000000) >> 24
            toc_entries.append(
                TocEntry(
                    seg_id,
                    seg_offset,
                    seg_len,
                    seg_attr,
                    DATA_SEGMENT_TYPES[seg_type - 1],
                )
            )

    return toc_entries


def read_segment(path: Path, toc_entry_offset: int):
    with open(path, mode="rb") as jt:
        jt.seek(toc_entry_offset)
        GUID.from_bytes(jt)
        ds_type, ds_len = struct.unpack("ii", jt.read(8))

        if DATA_SEGMENT_TYPES[ds_type - 1].compression:
            if VERSION == JtVersion.V9d5:
                comp_flag, comp_len, comp_alg = struct.unpack("iiB", jt.read(9))
                comp_len -= 1
                if comp_flag == 2 and comp_alg == 2:
                    ds_bytes = ByteStream(zlib.decompress(jt.read(comp_len)))
                else:
                    ds_bytes = ByteStream(jt.read(comp_len))
            else:
                comp_flag, comp_len, comp_alg = struct.unpack("IiB", jt.read(9))
                comp_len -= 1
                if comp_flag == 3 and comp_alg == 3:
                    ds_bytes = ByteStream(lzma.decompress(jt.read(comp_len)))
                else:
                    ds_bytes = ByteStream(jt.read(comp_len))
        else:
            ds_bytes = ByteStream(jt.read(ds_len))

        if ds_type == LSG.SEGMENT_TYPE_ID:
            return read_lsg_segment(ds_bytes, version=VERSION)
        if ds_type == Metadata.SEGMENT_TYPE_ID:
            return read_metadata_segment(ds_bytes, VERSION)
        if ds_type in SHAPE_SEGMENT_TYPE_IDS:
            return read_shape_segment(ds_bytes, VERSION)
        return None


def extract_points_and_topology(shape_obj) -> tuple[np.ndarray | None, object | None]:
    if shape_obj is None:
        return None, None

    vertex_lod = getattr(shape_obj, "vertex_shape_LOD_data", None)
    topo_lod = getattr(vertex_lod, "topo_mesh_compressed_lod_data", None) if vertex_lod else None
    topo_rep = getattr(topo_lod, "topo_mesh_compressed_rep_data", None) if topo_lod else None
    if topo_rep is None:
        return None, None

    vertex_records = getattr(topo_rep, "topologically_compressed_vertex_records", None)
    coord_array = getattr(vertex_records, "compressed_vertex_coordinate_array", None) if vertex_records else None
    vertex_coordinates = getattr(coord_array, "vertex_coordinates", None) if coord_array else None
    if not vertex_coordinates or len(vertex_coordinates) < 3:
        return None, None

    points = np.column_stack(
        (
            np.asarray(vertex_coordinates[0], dtype=np.float64),
            np.asarray(vertex_coordinates[1], dtype=np.float64),
            np.asarray(vertex_coordinates[2], dtype=np.float64),
        )
    )
    return points, topo_rep


def triangulate_component(
    points: np.ndarray,
    vfm,
    dual_vertices: list[int],
) -> tuple[np.ndarray, np.ndarray] | None:
    max_vertex_id = len(points) - 1
    triangles: list[tuple[int, int, int]] = []
    used_vertex_ids: set[int] = set()
    seen_faces: set[tuple[int, int, int]] = set()

    for dual_vertex in dual_vertices:
        valence = vfm.valence(dual_vertex)
        if valence < 3:
            continue

        ring = [vfm.face(dual_vertex, slot) for slot in range(valence)]
        ring = [vertex_id for vertex_id in ring if 0 <= vertex_id <= max_vertex_id]
        if len(ring) < 3:
            continue

        base = ring[0]
        used_vertex_ids.update(ring)
        for ring_index in range(1, len(ring) - 1):
            tri = (base, ring[ring_index], ring[ring_index + 1])
            if len({tri[0], tri[1], tri[2]}) < 3:
                continue
            tri_key = tuple(sorted(tri))
            if tri_key in seen_faces:
                continue
            seen_faces.add(tri_key)
            triangles.append(tri)

    if not triangles or not used_vertex_ids:
        return None

    ordered_vertex_ids = sorted(used_vertex_ids)
    remap = {vertex_id: index for index, vertex_id in enumerate(ordered_vertex_ids)}
    compact_points = points[ordered_vertex_ids]
    compact_triangles = np.asarray(
        [[remap[a], remap[b], remap[c]] for a, b, c in triangles],
        dtype=np.int32,
    )
    return compact_points, compact_triangles


def decode_shape_components(
    shape_offset: int,
    object_id: int,
    points: np.ndarray,
    topo_rep,
) -> list[DecodedMesh]:
    extent = float(np.ptp(points, axis=0).max())
    if 0 < extent < 1e-3:
        points = points * (1.0 / extent)

    def run_codec(max_components: int | None):
        codec = _MeshCodec(MeshCoderDriver(topo_rep))
        vfm = codec.run(max_components=max_components)
        _face_component, vertex_component = codec.component_maps()
        return vfm, vertex_component

    try:
        vfm, vertex_component = run_codec(max_components=None)
    except RuntimeError:
        try:
            vfm, vertex_component = run_codec(max_components=1)
        except RuntimeError as exc:
            logger.warning(
                "Skipping shape %s object %s because surface decode failed: %s",
                shape_offset,
                object_id,
                exc,
            )
            return []

    if vertex_component:
        component_ids = sorted(
            component_id
            for component_id in set(vertex_component.values())
            if component_id >= 0
        )
    else:
        component_ids = [0]
        vertex_component = {dual_vertex: 0 for dual_vertex in range(vfm.numVts())}

    decoded_meshes: list[DecodedMesh] = []
    for component_id in component_ids:
        dual_vertices = [
            dual_vertex
            for dual_vertex, mapped_component in vertex_component.items()
            if mapped_component == component_id
        ]
        triangulated = triangulate_component(points, vfm, dual_vertices)
        if triangulated is None:
            continue

        component_points, component_triangles = triangulated
        decoded_meshes.append(
            DecodedMesh(
                name=f"shape_{shape_offset}_object_{object_id}_component_{component_id}",
                points=component_points,
                triangles=component_triangles,
            )
        )

    return decoded_meshes


def load_decoded_meshes(path: Path) -> list[DecodedMesh]:
    decoded_meshes: list[DecodedMesh] = []
    toc_entries = read_table_of_contents(path)

    for entry in toc_entries:
        if entry.type.id not in SHAPE_SEGMENT_TYPE_IDS:
            continue

        logger.info("reading shape segment at offset %s", entry.offset)
        shape_segment = read_segment(path, entry.offset)
        if not isinstance(shape_segment, dict):
            continue

        for object_id, shape_obj in shape_segment.items():
            points, topo_rep = extract_points_and_topology(shape_obj)
            if points is None or topo_rep is None:
                continue

            decoded_meshes.extend(
                decode_shape_components(entry.offset, int(object_id), points, topo_rep)
            )

    return decoded_meshes


def default_output_path(jt_path: Path) -> Path:
    return jt_path.with_name(f"{jt_path.stem}_omniverse_meshes.txt")


def export_meshes_txt(meshes: list[DecodedMesh], jt_path: Path, output_path: Path) -> None:
    payload = {
        "format": "omniverse_mesh_export_v1",
        "source_jt": str(jt_path),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mesh_count": len(meshes),
        "meshes": [],
    }

    for mesh in meshes:
        payload["meshes"].append(
            {
                "name": mesh.name,
                "points": mesh.points.tolist(),
                "triangleVertexIndices": mesh.triangles.tolist(),
            }
        )

    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def visualize_meshes(meshes: list[DecodedMesh]) -> None:
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("open3d is required for --visualize") from exc

    for mesh in meshes:
        triangle_mesh = o3d.geometry.TriangleMesh()
        triangle_mesh.vertices = o3d.utility.Vector3dVector(mesh.points)
        triangle_mesh.triangles = o3d.utility.Vector3iVector(mesh.triangles)
        triangle_mesh.compute_vertex_normals()

        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=mesh.name)
        vis.add_geometry(triangle_mesh)
        vis.reset_view_point(True)
        vis.run()
        vis.destroy_window()


def main():
    parser = argparse.ArgumentParser(description="Decode a JT file into an Omniverse mesh export")
    parser.add_argument("path", help="Path to the JT file")
    parser.add_argument(
        "--output-txt",
        help="Where to write the Omniverse mesh export text file",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Open decoded meshes in Open3D after exporting",
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging_config.configure_logging(args.debug)
    if args.debug:
        sys.excepthook = debug_info

    jt_path = Path(args.path).resolve()
    output_path = Path(args.output_txt).resolve() if args.output_txt else default_output_path(jt_path)

    logger.info("Started")
    meshes = load_decoded_meshes(jt_path)
    if not meshes:
        raise RuntimeError(f"No decodable meshes found in {jt_path}")

    export_meshes_txt(meshes, jt_path, output_path)
    logger.info("Wrote %s meshes to %s", len(meshes), output_path)
    print(output_path)

    if args.visualize:
        visualize_meshes(meshes)

    logger.info("Finished")


if __name__ == "__main__":
    main()
