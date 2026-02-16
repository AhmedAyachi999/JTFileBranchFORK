import logging
from dataclasses import dataclass

from lsg.elementHeader import ElementHeader
from shape.triStripSetShapeLODElement import TriStripSetShapeLodElement
from lsg.types import JtVersion

logger = logging.getLogger(__name__)


@dataclass
class Shape:
    SEGMENT_TYPE_ID = 6


def read_shape_segment(ds_bytes, version=JtVersion.V9d5):
    end_of_elements = False
    i = 0
    shapes = {}
    while ds_bytes.remaining() > 0 and not end_of_elements:
        e_header = ElementHeader.from_bytes(ds_bytes)
        if e_header.object_type_id == TriStripSetShapeLodElement.TYPE_ID:
            pre_offset = ds_bytes.offset

            bs = ds_bytes.bytes[ds_bytes.offset -
                                25:ds_bytes.offset+e_header.length]
            logger.debug(bs.hex(' '))

            shape = TriStripSetShapeLodElement.from_bytes(
                ds_bytes, header=e_header, version=version)
            post_offset = ds_bytes.offset
            length = post_offset - pre_offset
            remaining = e_header.length - 25 - length
            ds_bytes.read(remaining)
            ds_bytes.read(4)
        elif e_header.object_type_id == ElementHeader.END_OF_ELEMENTS:
            shape = None
            end_of_elements = True
        else:
            shape = None
            logger.warning(
                f'Found unsupported element type {e_header.object_type_id}.')
            ds_bytes.read(e_header.length - 21)
        shapes[e_header.object_id] = shape
    return shapes


class ShapeLod0(Shape):
    SEGMENT_TYPE_ID = 7


class ShapeLod1(Shape):
    SEGMENT_TYPE_ID = 8
