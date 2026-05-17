import enum
import logging
import math
import struct

from codec.bitlength import decode_bitlength2, decode_bitlength3
from codec.codecDriver import CodecDriver
from codec.arithmetic import ProbabilityContext, decode_arithmetic, decode_arithmetic_v10
from lsg.types import JtVersion
from util.bitBuffer import BitBuffer
from util.byteStream import ByteStream

logger = logging.getLogger(__name__)


class PredictorType(enum.Enum):
    PredLag1 = 0
    PredLag2 = 1
    PredStride1 = 2
    PredStride2 = 3
    PredStripIndex = 4
    PredRamp = 5
    PredXor1 = 6
    PredXor2 = 7
    PredNULL = 8


class I32CDP2:
    CODECTYPE_NULL = 0
    CODECTYPE_BITLENGTH = 1
    CODECTYPE_ARITHMETIC = 3
    CODECTYPE_CHOPPER = 4
    CODECTYPE_MOVE_TO_FRONT = 5

    @classmethod
    def read_vec_i_32(
        cls,
        e_bytes: ByteStream,
        predictor_type: PredictorType = PredictorType.PredNULL,
        version=JtVersion.V9d5,
    ) -> [int]:
        decoded_symbols = cls.decode_bytes(e_bytes, version=version)
        unpacked_symbols = cls.unpack_residuals(decoded_symbols, predictor_type)
        return unpacked_symbols

    @classmethod
    def unpack_residuals(cls, residuals: [int], predictor_type: PredictorType) -> [int]:
        unpacked = []
        for i, res in enumerate(residuals):
            if predictor_type == PredictorType.PredNULL:
                unpacked.append(res)
            else:
                if i < 4:
                    unpacked.append(res)
                else:
                    predicted = cls.predict_value(unpacked, i, predictor_type)
                    if predictor_type in [
                        PredictorType.PredXor1,
                        PredictorType.PredXor2,
                    ]:
                        unpacked.append(res ^ predicted)
                    else:
                        unpacked.append(res + predicted)
        return unpacked

    @classmethod
    def predict_value(
        cls, values: [int], index: int, predictor_type: PredictorType
    ) -> int:
        v1 = values[index - 1]
        v2 = values[index - 2]
        v4 = values[index - 4]

        if predictor_type in [PredictorType.PredXor1, PredictorType.PredLag1]:
            return v1
        elif predictor_type in [PredictorType.PredXor2, PredictorType.PredLag2]:
            return v2
        elif predictor_type == PredictorType.PredStride1:
            return v1 + (v1 - v2)
        elif predictor_type == PredictorType.PredStride2:
            return v2 + (v2 - v4)
        elif predictor_type == PredictorType.PredStripIndex:
            if -8 < (v2 - v4) < 8:
                return v2 + (v2 - v4)
            else:
                return v2 + 2
        elif predictor_type == PredictorType.PredRamp:
            return index

    @classmethod
    def _move_to_front_decode(cls, values: list[int], offsets: list[int]) -> list[int]:
        window = []
        decoded_symbols = []
        value_index = 0
        for offset in offsets:
            if offset == -1:
                value = values[value_index]
                value_index += 1
                window.insert(0, value)
                if len(window) > 16:
                    window.pop()
                decoded_symbols.append(value)
            else:
                value = window[offset]
                del window[offset]
                window.insert(0, value)
                decoded_symbols.append(value)
        return decoded_symbols

    @classmethod
    def decode_bytes(cls, e_bytes: ByteStream, version=JtVersion.V9d5) -> list:
        logger.debug(
            f"starting decode of {e_bytes.bytes[e_bytes.offset:e_bytes.offset+40].hex(' ')} ..."
        )
        is_v10 = version == JtVersion.V10d5
        value_count = struct.unpack("<i", e_bytes.read(4))[0]
        if value_count <= 0:
            return []

        codec_type = struct.unpack("<B", e_bytes.read(1))[0]
        supported_codec_types = {
            cls.CODECTYPE_NULL,
            cls.CODECTYPE_BITLENGTH,
            cls.CODECTYPE_ARITHMETIC,
            cls.CODECTYPE_CHOPPER,
        }
        if is_v10:
            supported_codec_types.add(cls.CODECTYPE_MOVE_TO_FRONT)
        if codec_type not in supported_codec_types:
            raise RuntimeError(
                f"Codec Type {codec_type} not supported for {cls.__name__}"
            )
        elif codec_type == cls.CODECTYPE_CHOPPER:
            logger.debug(f"{codec_type=}")
            chop_bits = struct.unpack("<B", e_bytes.read(1))[0]
            if chop_bits == 0 and not is_v10:
                return cls.decode_bytes(e_bytes, version=version)
            value_bias, value_span_bits = struct.unpack("<iB", e_bytes.read(5))
            chopped_msb_data = cls.decode_bytes(e_bytes, version=version)
            chopped_lsb_data = cls.decode_bytes(e_bytes, version=version)

            decoded_symbols = []
            for msb, lsb in zip(chopped_msb_data, chopped_lsb_data):
                decoded_symbols.append(
                    (lsb | msb << (value_span_bits - chop_bits)) + value_bias
                )
            return decoded_symbols
        elif codec_type == cls.CODECTYPE_MOVE_TO_FRONT:
            logger.debug(f"{codec_type=}")
            values = cls.decode_bytes(e_bytes, version=version)
            offsets = cls.decode_bytes(e_bytes, version=version)
            return cls._move_to_front_decode(values, offsets)
        elif codec_type == cls.CODECTYPE_NULL:
            logger.debug(f"{codec_type=}")
            length = struct.unpack("<i", e_bytes.read(4))[0]
            decoded_symbols = struct.unpack(
                "<" + ("i" * (length // 4)), e_bytes.read(length)
            )
            return list(decoded_symbols)
        code_text_length = struct.unpack("<i", e_bytes.read(4))[0]
        words_to_read = int(math.ceil(code_text_length / 32.0))
        code_text = []
        for i in range(words_to_read):
            buffer = e_bytes.read(4)
            code_text.append(buffer[3])
            code_text.append(buffer[2])
            code_text.append(buffer[1])
            code_text.append(buffer[0])
        code_text = bytes(code_text)
        # code_text_words = list(struct.unpack("<" + "i" * words_to_read, code_text))

        int_32_probability_contexts = ProbabilityContext(0, 0, [])
        out_of_band_values = []
        if codec_type == cls.CODECTYPE_ARITHMETIC:
            logger.debug(f"{codec_type=}")
            int_32_probability_contexts = ProbabilityContext.from_bytes(
                e_bytes, version=version
            )
            if is_v10:
                if int_32_probability_contexts.has_out_of_band_values:
                    out_of_band_values = cls.decode_bytes(e_bytes, version=version)
            else:
                out_of_band_values = cls.decode_bytes(e_bytes, version=version)
            if code_text_length == 0 and len(out_of_band_values) == value_count:
                return out_of_band_values

        codec_driver = CodecDriver(
            code_text,
            code_text_length,
            value_count,
            int_32_probability_contexts,
            out_of_band_values,
        )
        codec_driver.bit_buffer = BitBuffer(
            ByteStream(code_text), endianness="big"
        )
        logger.debug(f"{codec_driver.bit_buffer.position=}")

        if codec_type == cls.CODECTYPE_BITLENGTH:
            logger.debug(f"{codec_type=}")
            logger.debug("calling decode_bitlength2 from i32csp2")
            decoded_symbols = (
                decode_bitlength3(codec_driver) if is_v10 else decode_bitlength2(codec_driver)
            )
        elif codec_type == cls.CODECTYPE_ARITHMETIC:
            logger.debug(f"{codec_type=}")
            decoded_symbols = (
                decode_arithmetic_v10(codec_driver)
                if is_v10
                else decode_arithmetic(codec_driver)
            )
        else:
            decoded_symbols = []

        if len(decoded_symbols) != value_count:
            logger.error(
                f"Codec produced wrong number of symbols: {len(decoded_symbols)} {value_count}"
            )
            # raise RuntimeError(f"Codec produced wrong number of symbols: {len(decoded_symbols)} {value_count}")
        return decoded_symbols
