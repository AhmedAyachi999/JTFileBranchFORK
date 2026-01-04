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
    total_bits = codec_driver.code_text_length
    logger.debug(f"{encoded_bits.position=}")

    # Handle fixed width
    if (mode := encoded_bits.read_int(1)) == 0:
        p = encoded_bits.position
        num_bits_from_min_symbol = encoded_bits.read_int(6)
        num_bits_from_max_symbol = encoded_bits.read_int(6)
        min_symbol = encoded_bits.read_signed_int(num_bits_from_min_symbol)
        max_symbol = encoded_bits.read_signed_int(num_bits_from_max_symbol)
        bit_width = get_bit_field_width(max_symbol - min_symbol)
        if (bit_width != 0):
            if ((total_bits - encoded_bits.position) / bit_width == expected_values):
                print("V9.5")
            else:
                encoded_bits.position = p
                logger.debug(f"{encoded_bits.position=}")
                logger.debug(
                    f"starting fixed length decode of {codec_driver.code_text.hex(' ')}"
                )
                db_str = " ".join([f"{int(b):08b}" for b in codec_driver.code_text])
                logger.debug(db_str)

                min_symbol = 0
                max_symbol = 0
                cNibbles = 0
                more_bits = True
                while more_bits:
                    tmp = encoded_bits.read_int(4)
                    min_symbol |= (tmp << (cNibbles * 4))
                    cNibbles += 1
                    more_bits = encoded_bits.read_int(1)
                cNibbles = 0
                more_bits = True
                while more_bits:
                    tmp = encoded_bits.read_int(4)
                    max_symbol |= (tmp << (cNibbles * 4))
                    cNibbles += 1
                    more_bits = encoded_bits.read_int(1)
                bit_width = (max_symbol - min_symbol).bit_length()
                logger.debug(f"{bit_width=} {min_symbol=} {max_symbol=}")
                logger.debug(f"{encoded_bits.position}")
                db_str = "".join([f"{int(b):08b}" for b in codec_driver.code_text])
                logger.debug(f"{db_str[encoded_bits.position:]=}, {total_bits=}")

                # Read each fixed-width field and output the value
        while (
                encoded_bits.position + bit_width < total_bits or len(decoded_symbols) < expected_values
        ):
            decoded_symbol = encoded_bits.read_int(bit_width)
            decoded_symbol += min_symbol
            decoded_symbols.append(decoded_symbol)
        logger.debug("finished fixed length decode")
        logger.debug(f"{decoded_symbols=}")
    # Handle variable width
    else:
        # Write out the mean value
        logger.debug(
            f"starting variable width decode of {codec_driver.code_text.hex(' ')}"
        )
        mean_value = encoded_bits.read_int(32)
        block_val_bits = encoded_bits.read_int(3)
        block_len_bits = encoded_bits.read_int(3)

        # Set the initial field-width
        max_field_decr = -(1 << (block_val_bits - 1))  # -ve number
        max_field_incr = (1 << (block_val_bits - 1)) - 1  # +ve number
        current_field_width = 0

        i = 0
        while i < expected_values:
            # Adjust the current field width to the target field width
            delta_field_width = encoded_bits.read_signed_int(block_val_bits)
            current_field_width += delta_field_width
            while (
                delta_field_width == max_field_decr
                or delta_field_width == max_field_incr
            ):
                delta_field_width = encoded_bits.read_signed_int(block_val_bits)
                current_field_width += delta_field_width

            # Read in the run length
            run_length = encoded_bits.read_int(block_len_bits)

            # Read in the data bits for the run
            for j in range(i, i + run_length):
                decoded_symbols.append(
                    encoded_bits.read_signed_int(current_field_width) + mean_value
                )

            # Advance to the end of the run
            i += run_length
        logger.debug("finished variable length debug")
    if encoded_bits.position != total_bits or len(decoded_symbols) != expected_values:
        logger.error(
            f"{encoded_bits.position=} {total_bits=} {len(decoded_symbols)=} {expected_values=} with {mode=}"
        )
        # raise ValueError("BitlengthCodec2 didn't consume all bits!")
    else:
        logger.debug(f"Decode worked with {mode=}")
    return decoded_symbols
