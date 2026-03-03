import logging
from codec.codecDriver import CodecDriver

logger = logging.getLogger(__name__)


def get_bit_field_width(symbol: int):
    symbol = abs(symbol)
    if symbol == 0:
        return 0
    i = 1
    bit_field_width = 0
    while i <= symbol and bit_field_width < 31:
        i += i
        bit_field_width += 1
    return bit_field_width


def decode_bitlength2(codec_driver: CodecDriver):
    encoded_bits = codec_driver.bit_buffer
    decoded_symbols = []

    expected_values = codec_driver.value_count
    logger.debug(f"{encoded_bits.position=}")

    # Helper lambdas mirroring C++ ReadU32Or0 / ReadI32Or0
    read_u32_or0 = lambda n: encoded_bits.read_int(n) if n else 0
    read_i32_or0 = lambda n: encoded_bits.read_signed_int(n) if n else 0

    mode = encoded_bits.read_int(1)
    if mode == 0:
        # Fixed width (C++ decodeFixedWidth)
        min_bits = encoded_bits.read_int(6)
        max_bits = encoded_bits.read_int(6)
        min_symbol = read_i32_or0(min_bits)
        max_symbol = read_i32_or0(max_bits)
        a_range = max_symbol - min_symbol
        if a_range <= 0:
            while len(decoded_symbols) < expected_values:
                decoded_symbols.append(min_symbol)
        else:
            field_width = 1
            tmp = a_range >> 1
            while tmp:
                field_width += 1
                tmp >>= 1
            while len(decoded_symbols) < expected_values:
                decoded_symbols.append(encoded_bits.read_int(field_width) + min_symbol)
        logger.debug("finished fixed length decode")
    else:
        # Variable width (C++ decodeVariableWidth)
        mean_value = encoded_bits.read_signed_int(32)
        chg_width_bits = encoded_bits.read_int(3)
        run_len_bits = encoded_bits.read_int(3)
        max_decr = -(1 << (chg_width_bits - 1))
        max_incr = (1 << (chg_width_bits - 1)) - 1
        field_width = 0
        while len(decoded_symbols) < expected_values:
            delta = encoded_bits.read_signed_int(chg_width_bits)
            field_width += delta
            while delta == max_decr or delta == max_incr:
                delta = encoded_bits.read_signed_int(chg_width_bits)
                field_width += delta
            run_len = encoded_bits.read_int(run_len_bits)
            if field_width > 0:
                for _ in range(run_len):
                    decoded_symbols.append(encoded_bits.read_signed_int(field_width) + mean_value)
            else:
                for _ in range(run_len):
                    decoded_symbols.append(mean_value)
        logger.debug("finished variable length decode")

    if len(decoded_symbols) != expected_values:
        logger.error(
            f"{len(decoded_symbols)=} {expected_values=} with {mode=}"
        )
    return decoded_symbols
