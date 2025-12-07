# import knime.scripting.io as knio
import dataclasses
import argparse
import logging
from typing import Final
import struct
from dataclasses import dataclass, field
import zlib
import lzma
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# matplotlib.use('tkagg')

import pandas as pd
from core import logging_config
from lsg.lsg import LSG, read_lsg_segment
from lsg.types import GUID, JtVersion
from metadata.metadata import Metadata, read_metadata_segment
from shape.shape import Shape, read_shape_segment, ShapeLod0, ShapeLod1
from shape.topologicallyCompressedRepData import MeshDecoder
from util.byteStream import ByteStream

import traceback
import sys


def debug_info(type, value, tb):
    import pdb
    traceback.print_exception(type, value, tb)
    pdb.post_mortem(tb)



PATH = ""
PATH_9: Final = r'8W8_827_605____PCA_TM__010_____HALTER_HKL_________150819______________.jt'
PATH_10: Final = r'83H_837_461____PCA_TM__400_____HEBEGESTELL_VT_____HI_LEX_20220722_____.jt'
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
    # V10.5 below
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


def read_table_of_contents(path: str):
    """
    opens jt file specified in PATH and reads table of contents
    Returns:
        list[ToxEntry]: test
    """
    with open(path, mode='rb') as jt:
        global VERSION
        # read header
        jt_version = jt.read(80)
        jt_version = jt_version[8:12]
        if jt_version == b'9.5 ' or jt_version == b'9.4 ':
            VERSION = JtVersion.V9d5
        #TODO: Check differences between 10.5 and 10.0
        elif jt_version == b'10.5' or jt_version == b"10.0":
            VERSION = JtVersion.V10d5
        else:
            raise NotImplementedError(f"version {jt_version} not supported")
        logger.info(f"reading jt file with version {VERSION}")
        # TODO: check byteorder
        jt_bo = int.from_bytes(jt.read(1), byteorder="little")
        jt_reserved = jt.read(4)
        jt_toc_offset = struct.unpack("i", jt.read(4))[0]

        # read toc
        jt.seek(jt_toc_offset)
        jt_toc_entry_count = int.from_bytes(jt.read(4), byteorder="little")
        jt_toc_entry = []
        for i in range(jt_toc_entry_count):
            seg_id = GUID.from_bytes(jt)
            # TODO: actual version parsing
            if VERSION == JtVersion.V9d5:
                seg_offset, seg_len, seg_attr = struct.unpack(
                    "iiI", jt.read(12))
            elif VERSION == JtVersion.V10d5:
                seg_offset, seg_len, seg_attr = struct.unpack(
                    "QII", jt.read(16))

            seg_type = seg_attr & 0xFF000000
            seg_type = seg_type >> 24
            jt_toc_entry.append(TocEntry(
                seg_id, seg_offset, seg_len, seg_attr, DATA_SEGMENT_TYPES[seg_type - 1]))
    return jt_toc_entry


def read_segment(path, toc_entry_offset: int):
    with open(path, mode='rb') as jt:
        jt.seek(toc_entry_offset)
        ds_id = GUID.from_bytes(jt)
        ds_type, ds_len = struct.unpack("ii", jt.read(8))

        if DATA_SEGMENT_TYPES[ds_type - 1].compression:
            # only first element in segment
            if VERSION == JtVersion.V9d5:
                comp_flag, comp_len, comp_alg = struct.unpack(
                    "iiB", jt.read(9))
                # comp_alg is part of comp_len
                comp_len -= 1
                if comp_flag == 2 and comp_alg == 2:
                    ds_bytes = ByteStream(zlib.decompress(jt.read(comp_len)))
                else:
                    ds_bytes = ByteStream(jt.read(comp_len))
            elif VERSION == JtVersion.V10d5:
                comp_flag, comp_len, comp_alg = struct.unpack(
                    "IiB", jt.read(9))
                comp_len -= 1
                if comp_flag == 3 and comp_alg == 3:
                    ds_bytes = ByteStream(lzma.decompress(jt.read(comp_len)))
                else:
                    ds_bytes = ByteStream(jt.read(comp_len))
        else:
            ds_bytes = ByteStream(jt.read(ds_len))
        if ds_type == LSG.SEGMENT_TYPE_ID:
            return read_lsg_segment(ds_bytes, version=VERSION)
        elif ds_type == Metadata.SEGMENT_TYPE_ID:
            return read_metadata_segment(ds_bytes, VERSION)
        elif ds_type == Shape.SEGMENT_TYPE_ID or ds_type == ShapeLod1.SEGMENT_TYPE_ID:
            return read_shape_segment(ds_bytes, VERSION)


def toc_entries_to_df(toc_entries: list[TocEntry]):
    return pd.DataFrame(flatten_toc(toc_entries))


def flatten_toc(toc_entries: list[TocEntry]):
    """
    flattens list of TocEntry to list of dicts with depth 1 for tabular representation
    Args:
        toc_entries: list of TocEntrys

    Returns:
        list[dict]: dictionaries containing important toc information
    """
    for e in toc_entries:
        yield {
            "GUID": str(e.guid),
            "Offset": e.offset,
            "Segment Type ID": e.type.id,
            "Segment Type Name": e.type.name,
            "Segment Compression": e.type.compression
        }


def flatten_lsg_nodes(lsg_nodes: list):
    for node in lsg_nodes:
        data = dataclasses.asdict(node)
        del data["element_header"]
        child_nodes = ""
        if node.element_header.object_base_type == 1:
            child_nodes = node.group_node_data.child_node_object_id
        elif node.element_header.object_base_type == 5:
            data = data["value"]
        elif node.element_header.object_base_type == 8:
            del data["base_property_atom_data"]
        yield {
            "object id": str(node.element_header.object_id),
            "object type name": str(type(node).__name__),
            "object type id": str(node.element_header.object_type_id),
            "object base type": str(node.element_header.object_base_type),
            "child node ids": str(child_nodes),
            "val": str(data)
        }


def main():
    parser = argparse.ArgumentParser(description="Load a jt file")
    # parser.add_argument('version', metavar='v', type=int,
                        # nargs='?', help='version to load', default=10)
    parser.add_argument("path")
    parser.add_argument('--spin', action="store_true",
                        help="spin the 3D view (interactive backends only)")
    parser.add_argument('--debug', action="store_true")
    args = parser.parse_args()
    logging_config.configure_logging(args.debug)
    if args.debug:
        sys.excepthook = debug_info
    logger.info("Started")
    logger.debug("showing debug information")

    # if args.version == 9:
    #     PATH = PATH_9
    # else:
    #     PATH = PATH_10
    PATH = args.path

    jt_toc = read_table_of_contents(PATH)

    def is_lsg(toc_entry: TocEntry):
        return toc_entry.type.id == 1

    def is_shape(toc_entry: TocEntry):
        return toc_entry.type.id == 6

    lsg_entry = list(filter(is_lsg, jt_toc))
    lsg_entry = lsg_entry[0]

    shape_entries = list(filter(is_shape, jt_toc))
    logger.info(f"starting to read lsg segment at {lsg_entry.offset}")
    lsg = read_segment(PATH, lsg_entry.offset)
    logger.info(f"finished read of lsg segment {lsg}")
    print(lsg.ascii_lsg_tree())

    guid_shape_in_lsg = (
        GUID((0x0C06BE92, 0x467A, 0x11E5, 0x80, 0,
             0x91, 0x9d, 0x66, 0x7e, 0x24, 0x30)),
        GUID((0x0C06BE99, 0x467A, 0x11E5, 0x80, 0,
             0x91, 0x9d, 0x66, 0x7e, 0x24, 0x30)),
        GUID((0x0C06BE9e, 0x467A, 0x11E5, 0x80, 0,
             0x91, 0x9d, 0x66, 0x7e, 0x24, 0x30))
    )
    shapes = []
    print(shape_entries)
    # shape_entries = filter(lambda se: se.guid in guid_shape_in_lsg, shape_entries)
    for entry in shape_entries:
        logger.info(
            f"starting to read shape segment {entry.guid} at {entry.offset}")
        shapes.append(read_segment(PATH, entry.offset))
        logger.info(f"finished reading shape segment at {entry.offset}")
    print(shapes)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    def log_shape_streams(idx, rep_data):
        face_deg_streams = rep_data.face_degrees or []
        ctx_lengths = [len(face_deg_streams[i]) if i < len(face_deg_streams) else 0 for i in range(8)]
        print(f"[Shape {idx}] valences={len(rep_data.vertex_valences or [])} "
              f"face_degrees_per_ctx={ctx_lengths} "
              f"split_faces={len(rep_data.split_face_syms or [])} "
              f"split_pos={len(rep_data.split_face_positions or [])}")
        if any(l == 0 for l in ctx_lengths):
            print(f"[Shape {idx}] Red flag: some degree context streams are empty.")

    # Decode each shape, print what happens, and scatter points
    for i, sh in enumerate(shapes):
        lod0 = sh.get(0)
        if lod0 is None:
            continue
        rep_data = (
            lod0.vertex_shape_LOD_data.topo_mesh_compressed_lod_data
            .topo_mesh_compressed_rep_data
        )
        coord_arr = rep_data.topologically_compressed_vertex_records.compressed_vertex_coordinate_array
        coords = coord_arr.vertex_coordinates  # (3, N)
        if coords is None or len(coords) != 3:
            print(f"[Shape {i}] Red flag: missing coordinate array, skipping shape")
            continue

        x, y, z = coords
        ax.scatter(x, y, z, s=2)

        log_shape_streams(i, rep_data)
        try:
            decoded = MeshDecoder(rep_data).decode()
        except Exception as exc:
            print(f"[Shape {i}] ERROR during topology decode: {exc} (continuing)")
            # Fallback: link points sequentially so we still see something
            if len(x) > 1:
                xs = list(x)
                ys = list(y)
                zs = list(z)
                ax.plot(xs, ys, zs, linewidth=0.5, alpha=0.6, color="gray")
            continue

        faces = decoded.face_vertices
        print(f"[Shape {i}] Decoded {len(faces)} faces, {decoded.vertex_count} vertices")
        for j, face in enumerate(faces[:5]):
            print(f"  Face {j}: {face}")

        # Plot decoded faces as filled polys (for better shape perception)
        n_verts = len(x)
        polys = []
        for face in faces:
            if not face or any((v < 0 or v >= n_verts) for v in face):
                continue
            verts = [(x[idx], y[idx], z[idx]) for idx in face]
            polys.append(verts)
        if polys:
            coll = Poly3DCollection(polys, alpha=0.25, linewidths=0.2, edgecolors="k")
            coll.set_facecolor((0.2, 0.6, 0.8, 0.3))
            ax.add_collection3d(coll)

    if args.spin:
        backend = plt.get_backend().lower()
        if "agg" in backend:
            print("Spin requested but backend is non-interactive (Agg); skipping spin.")
        else:
            for az in range(0, 360, 2):
                ax.view_init(elev=30, azim=az)
                plt.draw()
                plt.pause(0.01)

    # Ensure mouse interactions stay active; map left-drag=rotate, right-drag=pan (disable zoom remap)
    try:
        ax.mouse_init(rotate_btn=1, pan_btn=3, zoom_btn=None)
    except Exception:
        pass

    plt.show()
    logger.info("Finished")


if __name__ == "__main__":
    main()
