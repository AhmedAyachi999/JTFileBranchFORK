import logging
import struct
import sys
from dataclasses import dataclass

from .metaDataNodeElement import MetaDataNodeElement
from .partNodeElement import PartNodeElement
from .rangeLODNodeElement import RangeLODNodeElement
from .groupNodeElement import GroupNodeElement
from .triStripSetShapeNodeElement import TriStripSetShapeNodeElement
from .materialAttributeElement import MaterialAttributeElement
from .pointSetShapeNodeElement import PointSetShapeNodeElement
from .polylineSetShapeNodeElement import PolylineSetShapeNodeElement
from .partitionNodeElement import PartitionNodeElement
from .instanceNodeElement import InstanceNodeElement
from .elementHeader import ElementHeader
from .lsgNode import LSGNode
from .dummyNodeElement import DummyNodeElement
from properties import (
    LateLoadedPropertyAtom,
    FloatingPointPropertyAtom,
    StringPropertyAtom,
)
from properties.dummyPropertyAtom import DummyPropertyAtom
from .types import JtVersion

logger = logging.getLogger(__name__)


@dataclass
class LSG:
    """
    Class representing data contained in the LSG data segment of a .jt-file.
    """

    SEGMENT_TYPE_ID = 1
    nodes: dict[int, LSGNode]
    props: dict
    property_table: list

    rootNode: LSGNode

    def __init__(self, nodes: dict, props: dict, property_table: list):
        self.nodes = nodes
        self.props = props
        self.property_table = property_table
        self._build_lsg()
        self._set_node_properties()

    def _build_lsg(self):
        self.rootNode = self.nodes[0]
        for k_n in self.nodes.keys():
            node = self.nodes[k_n]
            node.child_nodes = [self.nodes[c] for c in node.child_node_object_id]
            node.attributes = [self.nodes[a] for a in node.attr_object_id]
            node.properties = {}

    def _set_node_properties(self):
        for row in self.property_table:
            object_id = row["object id"]
            key_id = row["key property atom"]
            val_id = row["value property atom"]
            if self.props != None:
                key = self.props[key_id].val
                val = self.props[val_id].val
            else:
                val = -1
                # TODO: DO SOMETHING FOR V10.5
            self.nodes[object_id].properties[key] = val

    def ascii_lsg_tree(self, max_depth: int = 50):
        def string_for_node(node: LSGNode, depth=0, seen=None):
            if seen is None:
                seen = set()
            node_str = "  " + depth * "|     " + "|-- " + f"{str(node)}"
            if depth >= max_depth:
                return node_str + " [max_depth]"
            node_id = getattr(node, "object_id", None)
            if node_id is not None:
                if node_id in seen:
                    return node_str + " [cycle]"
                seen.add(node_id)
            if len(node.child_node_object_id) == 0:
                return node_str
            child_str = [string_for_node(c, depth + 1, seen.copy()) for c in node.child_nodes]
            child_str = "".join((f"{cs}" for cs in child_str))
            return f"{node_str}\n" + child_str

        return string_for_node(self.rootNode)


def read_lsg_nodes(ds_bytes, version=JtVersion.V9d5) -> dict:
    element_idx = 0
    end_of_elements = False
    nodes = {}
    while ds_bytes.remaining() > 0 and not end_of_elements:
        e_header = ElementHeader.from_bytes(ds_bytes)

        # save offset/ read head position to realign after element read
        read_head = ds_bytes.offset
        logger.debug(f"node {element_idx} has type {e_header.object_type_id}")
        if e_header.object_type_id == PartitionNodeElement.TYPE_ID:
            start = ds_bytes.offset
            node_data = PartitionNodeElement.from_bytes(
                ds_bytes, header=e_header, version=version
            )
            end = ds_bytes.offset
            remaining = (end - start) - e_header.length
            if remaining > 0:
                logger.debug(f"{remaining} bytes left for node, reading and discarding")
                ds_bytes.read(remaining)
        elif e_header.object_type_id == MetaDataNodeElement.TYPE_ID:
            node_data = MetaDataNodeElement.from_bytes(
                ds_bytes, header=e_header, version=version
            )
        elif e_header.object_type_id == PartNodeElement.TYPE_ID:
            node_data = PartNodeElement.from_bytes(
                ds_bytes, header=e_header, version=version
            )
        elif e_header.object_type_id == RangeLODNodeElement.TYPE_ID:
            node_data = RangeLODNodeElement.from_bytes(
                ds_bytes, header=e_header, version=version
            )
        elif e_header.object_type_id == GroupNodeElement.TYPE_ID:
            node_data = GroupNodeElement.from_bytes(
                ds_bytes, header=e_header, version=version
            )
        elif e_header.object_type_id == InstanceNodeElement.TYPE_ID:
            node_data = InstanceNodeElement.from_bytes(
                ds_bytes, header=e_header, version=version
            )
        elif e_header.object_type_id == TriStripSetShapeNodeElement.TYPE_ID:
            node_data = TriStripSetShapeNodeElement.from_bytes(
                ds_bytes, header=e_header, version=version
            )
        elif e_header.object_type_id == MaterialAttributeElement.TYPE_ID:
            node_data = MaterialAttributeElement.from_bytes(
                ds_bytes, header=e_header, version=version
            )
        elif e_header.object_type_id == PointSetShapeNodeElement.TYPE_ID:
            node_data = PointSetShapeNodeElement.from_bytes(
                ds_bytes, header=e_header, version=version
            )
        elif e_header.object_type_id == PolylineSetShapeNodeElement.TYPE_ID:
            node_data = PolylineSetShapeNodeElement.from_bytes(
                ds_bytes, header=e_header, version=version
            )
        elif e_header.object_type_id == ElementHeader.END_OF_ELEMENTS:
            logger.debug("End of Elements reached")
            end_of_elements = True
            node_data = None
        else:
            logger.warning(
                f"Found unsupported element type {e_header.object_type_id} while reading LSG segment"
            )
            # ds_bytes.read(e_header.length - 21)
            node_data = DummyNodeElement(e_header)
        if node_data is not None:
            nodes[e_header.object_id] = node_data
            # realign read position to mitigate propagation of offset errors between elements
            ds_bytes.offset = read_head
            ds_bytes.read(e_header.length - 21)
        else:
            # realign read position to mitigate propagation of offset errors between elements
            ds_bytes.offset = read_head
            ds_bytes.read(e_header.length - 16)
        element_idx += 1
    return nodes


def read_lsg_props(ds_bytes, version=JtVersion.V9d5) -> dict:
    end_of_elements = False
    props = {}
    element_idx = 0
    while ds_bytes.remaining() > 0 and not end_of_elements:
        e_header = ElementHeader.from_bytes(ds_bytes)
        # save offset/ read head position to realign after element read
        read_head = ds_bytes.offset
        prop = None
        logger.debug(f"prop {element_idx} has type {e_header.object_type_id}")
        if e_header.object_type_id == LateLoadedPropertyAtom.TYPE_ID:
            prop = LateLoadedPropertyAtom.from_bytes(
                ds_bytes, header=e_header, version=version
            )
        elif e_header.object_type_id == FloatingPointPropertyAtom.TYPE_ID:
            prop = FloatingPointPropertyAtom.from_bytes(
                ds_bytes, header=e_header, version=version
            )
        elif e_header.object_type_id == StringPropertyAtom.TYPE_ID:
            prop = StringPropertyAtom.from_bytes(
                ds_bytes, header=e_header, version=version
            )
        elif e_header.object_type_id == ElementHeader.END_OF_ELEMENTS:
            logger.debug("end of elements reached")
            end_of_elements = True
        else:
            prop = DummyPropertyAtom.from_bytes(None, header=e_header)
            logger.warning(f"element type {e_header.object_type_id} not implemented")
        if prop is not None:
            props[e_header.object_id] = prop
            # realign read position to mitigate propagation of offset errors between elements
            ds_bytes.offset = read_head
            ds_bytes.read(e_header.length - 21)
        else:
            # realign read position to mitigate propagation of offset errors between elements
            ds_bytes.offset = read_head
            ds_bytes.read(e_header.length - 16)
        element_idx += 1
    return props


def read_property_table(ds_bytes) -> list:
    pt_version_number, element_property_table_count = struct.unpack(
        "<hi", ds_bytes.read(6)
    )
    logger.debug(f"{pt_version_number=}, {element_property_table_count=}")
    property_table = []
    i = 1
    while ds_bytes.remaining() >= 12 and i < element_property_table_count:
        element_object_id = struct.unpack("i", ds_bytes.read(4))[0]
        has_keys = True
        while has_keys:
            key_property_atom_bytes = ds_bytes.read(4)
            if len(key_property_atom_bytes) != 4:
                has_keys = False
            else:
                key_property_atom = struct.unpack("i", key_property_atom_bytes)[0]
                if key_property_atom == 0:
                    has_keys = False
                else:
                    val_property_atom = struct.unpack("i", ds_bytes.read(4))[0]
                    property_table.append(
                        {
                            "object id": element_object_id,
                            "key property atom": key_property_atom,
                            "value property atom": val_property_atom,
                        }
                    )
        i += 1
    return property_table


def read_lsg_segment(ds_bytes, version=JtVersion.V9d5):
    # read data
    nodes = read_lsg_nodes(ds_bytes, version=version)
    props = read_lsg_props(ds_bytes, version=version)
    if JtVersion == JtVersion.V10d5:
        property_table = read_property_table(ds_bytes)
    else:
        property_table = []
    lsg = LSG(nodes, props, property_table)
    return lsg
