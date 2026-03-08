# import knime.scripting.io as knio
import dataclasses
import argparse
import logging
from typing import Final
import struct
from dataclasses import dataclass, field
import zlib
import lzma
import numpy as np
import open3d as o3d
# matplotlib.use('tkagg')

import pandas as pd
from core import logging_config
from lsg.lsg import LSG, read_lsg_segment
from lsg.types import GUID, JtVersion
from metadata.metadata import Metadata, read_metadata_segment
from shape.shape import (
    Shape,
    read_shape_segment,
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
)
from shape.topologicallyCompressedRepData import MeshCoderDriver, _MeshCodec
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
        elif ds_type in {
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
        }:
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

    shape_segment_type_ids = {
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

    def is_shape(toc_entry: TocEntry):
        return toc_entry.type.id in shape_segment_type_ids

    lsg_entry = list(filter(is_lsg, jt_toc))
    lsg_entry = lsg_entry[0]

    shape_entries = list(filter(is_shape, jt_toc))
    if not shape_entries:
        logger.warning("No shape segments found.")
        return
    logger.info(f"starting to read lsg segment at {lsg_entry.offset}")
    lsg = read_segment(PATH, lsg_entry.offset)
    logger.info(f"finished read of lsg segment {lsg}")
    print(lsg.ascii_lsg_tree())

    def try_extract_vertex_coords(shape_obj):
        if shape_obj is None:
            return None
        vertex_lod = getattr(shape_obj, "vertex_shape_LOD_data", None)
        if vertex_lod is None:
            return None
        topo_lod = getattr(vertex_lod, "topo_mesh_compressed_lod_data", None)
        if topo_lod is None:
            return None
        topo_rep = getattr(topo_lod, "topo_mesh_compressed_rep_data", None)
        if topo_rep is None:
            return None
        vtx_records = getattr(topo_rep, "topologically_compressed_vertex_records", None)
        if vtx_records is None:
            return None
        coord_array = getattr(vtx_records, "compressed_vertex_coordinate_array", None)
        if coord_array is None:
            return None
        coords = getattr(coord_array, "vertex_coordinates", None)
        if not coords or len(coords) < 3:
            return None
        x, y, z = coords[0], coords[1], coords[2]
        return x, y, z


    extracted_count = 0
    for entry in jt_toc:
        if not is_shape(entry):
            continue
        logger.info(
            f"starting to read shape segment {entry.guid} at {entry.offset}")
        shape_segment = read_segment(PATH, entry.offset)
        logger.info(f"finished reading shape segment at {entry.offset}")

        all_x, all_y, all_z = [], [], []
        if isinstance(shape_segment, dict):
            shape_iter = iter(shape_segment.items())
            first = next(shape_iter, None)
            if first is not None:
                object_id, shape_obj = first
                coords = try_extract_vertex_coords(shape_obj)
                if coords is not None:
                    x, y, z = coords
                    all_x.extend(x)
                    all_y.extend(y)
                    all_z.extend(z)
                    extracted_count += 1

        if all_x and all_y and all_z:
            points = np.column_stack((all_x, all_y, all_z))
            vertex_lod = getattr(shape_obj, "vertex_shape_LOD_data", None)
            topo_lod = getattr(vertex_lod, "topo_mesh_compressed_lod_data", None) if vertex_lod else None
            topo_rep = getattr(topo_lod, "topo_mesh_compressed_rep_data", None) if topo_lod else None
            if topo_rep is None:
                logger.warning(
                    f"No topologically compressed rep data for surface decode at offset {entry.offset}."
                )
                continue

            driver = MeshCoderDriver(topo_rep)
            codec = _MeshCodec(driver)
            try:
                vfm = codec.run()
            except RuntimeError as exc:
                logger.warning(f"Surface decode failed at offset {entry.offset}: {exc}")
                continue

            triangles = []
            max_vid = len(points) - 1
            for i_vtx in range(vfm.numVts()):
                valence = vfm.valence(i_vtx)
                if valence < 3:
                    continue
                ring = [vfm.face(i_vtx, slot) for slot in range(valence)]
                ring = [vid for vid in ring if 0 <= vid <= max_vid]
                if len(ring) < 3:
                    continue
                base = ring[0]
                for k in range(1, len(ring) - 1):
                    triangles.append([base, ring[k], ring[k + 1]])

            if not triangles:
                logger.warning(
                    f"No surface triangles reconstructed at offset {entry.offset}."
                )
                continue
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(points)
            mesh.triangles = o3d.utility.Vector3iVector(
                np.array(triangles, dtype=np.int32)
            )
            mesh.compute_vertex_normals()
            o3d.visualization.draw_geometries([mesh])
        else:
            logger.warning(
                f"No vertex coordinates found to plot for shape at offset {entry.offset}."
            )

    logger.info(
        f"Extracted vertex arrays from {extracted_count} shapes "
        f"across {len(shape_entries)} shape segments"
    )
    logger.info("Finished")


if __name__ == "__main__":
    main()
