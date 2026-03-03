import struct


class ByteStream:
    bytes: bytes
    offset: int

    def __init__(self, b, offset=0, debug_read=False):
        self.bytes = b
        self.offset = offset
        self.debug_read = debug_read

    def read(self, length: int):
        end = min(self.offset + length, len(self.bytes))
        data = self.bytes[self.offset:end]
        self.offset = end
        return data

    def get(self, index: int):
        self.offset = index + 1
        return self.bytes[self.offset-1]

    def seek(self, i: int):
        self.offset = i

    def remaining(self):
        return len(self.bytes) - self.offset


def read_vec_i_32(byte_stream):
    count = int.from_bytes(byte_stream.read(4), "little")
    return list(struct.unpack("<" + "i" * count, byte_stream.read(count * 4)))


def read_vec_u_32(byte_stream):
    count = int.from_bytes(byte_stream.read(4), "little")
    return list(struct.unpack("<" + "I" * count, byte_stream.read(count * 4)))
