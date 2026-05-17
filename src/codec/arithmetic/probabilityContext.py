from dataclasses import dataclass

from lsg.types import JtVersion
from util.bitBuffer import BitBuffer


@dataclass
class CntxEntry:
    symbol: int
    count: int
    value: int
    next_context: int = 0

    @classmethod
    def from_bits(
        cls,
        bits: BitBuffer,
        n_symbol_bits: int,
        n_occurrence_count_bits: int,
        n_value_bits: int,
        minimum_value: int,
        next_context_bits: int = -1,
    ):
        if n_symbol_bits >= 0:
            symbol = bits.read_int(n_symbol_bits) - 2
        else:
            symbol = 0 if bits.read_int(1) == 0 else -2
        count = bits.read_int(n_occurrence_count_bits)
        value = bits.read_int(n_value_bits) + minimum_value
        next_context = bits.read_int(next_context_bits) if next_context_bits >= 0 else 0
        return CntxEntry(symbol, count, value, next_context)


@dataclass
class ProbabilityContext:
    total_count: int
    num_entries: int
    table: list[CntxEntry]
    has_out_of_band_values: bool = False

    @classmethod
    def from_bytes(cls, e_bytes, version=JtVersion.V9d5):
        bit_buffer = BitBuffer(e_bytes, endianness="big")
        bit_buffer.position = e_bytes.offset << 3

        if version == JtVersion.V10d5:
            probability_context_table_entry_count = bit_buffer.read_int(16)
            num_symbol_bits = -1
            num_occurrence_count_bits = bit_buffer.read_int(6)
            number_value_bits = bit_buffer.read_int(7)
            min_value = bit_buffer.read_signed_int(32)
        else:
            probability_context_table_entry_count = bit_buffer.read_int(16)
            num_symbol_bits = bit_buffer.read_int(6)
            num_occurrence_count_bits = bit_buffer.read_int(6)
            number_value_bits = bit_buffer.read_int(6)
            min_value = bit_buffer.read_signed_int(32)

        probability_context_table_entries = []
        total_count = 0
        has_out_of_band_values = False
        for _ in range(probability_context_table_entry_count):
            entry = CntxEntry.from_bits(
                bit_buffer,
                num_symbol_bits,
                num_occurrence_count_bits,
                number_value_bits,
                min_value,
            )
            probability_context_table_entries.append(entry)
            total_count += entry.count
            if entry.symbol == -2:
                has_out_of_band_values = True

        e_bytes.offset = (bit_buffer.position + 7) >> 3
        return ProbabilityContext(
            total_count,
            probability_context_table_entry_count,
            probability_context_table_entries,
            has_out_of_band_values,
        )

    def accumulated_probability_counts(self):
        acc = 0
        entry_by_acc_count = {}
        for i, entry in enumerate(self.table):
            acc += entry.count
            entry_by_acc_count[acc - 1] = i
        return acc, entry_by_acc_count
