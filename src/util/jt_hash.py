import struct


def mix(a, b, c):
    """
     mix -- mix 3 32-bit values reversibly.
     
     For every delta with one or two bit set, and the deltas of all three
     high bits or all three low bits, whether the original value of a,b,c
     is almost all zero or is uniformly distributed,
     * If mix() is run forward or backward, at least 32 bits in a,b,c
     have at least 1/4 probability of changing.
     * If mix() is run forward, every bit of c will change between 1/3 and
     2/3 of the time. (Well, 22/100 and 78/100 for some 2-bit deltas.)
     mix() was built out of 36 single-cycle latency instructions in a
     structure that could supported 2x parallelism, like so:
     
     a -= b;
     a -= c; x = (c>>13);
     b -= c; a ^= x;
     b -= a; x = (a<<8);
     c -= a; b ^= x;
     c -= b; x = (b>>13);
     
     Unfortunately, superscalar Pentiums and Sparcs can't take advantage
     of that parallelism. They've also turned some of those single-cycle
     latency instructions into multi-cycle latency instructions. Still,
     this is the fastest good hash I could find. There were about 2^^68
     to choose from. I only looked at a billion or so.
    """
    a = (a - b) & 0xffffffff
    a = (a - c) & 0xffffffff
    a = (a ^ (c >> 13)) & 0xffffffff
    
    b = (b - c) & 0xffffffff
    b = (b - a) & 0xffffffff
    b = (b ^ (a << 8)) & 0xffffffff

    c = (c - a) & 0xffffffff
    c = (c - b) & 0xffffffff
    c = (c ^ (b >> 13)) & 0xffffffff

    a = (a - b) & 0xffffffff
    a = (a - c) & 0xffffffff
    a = (a ^ (c >> 12)) & 0xffffffff

    b = (b - c) & 0xffffffff
    b = (b - a) & 0xffffffff
    b = (b ^ (a << 16)) & 0xffffffff
    
    c = (c - a) & 0xffffffff
    c = (c - b) & 0xffffffff
    c = (c ^ (b >> 5)) & 0xffffffff

    a = (a - b) & 0xffffffff
    a = (a - c) & 0xffffffff
    a = (a ^ (c >> 3)) & 0xffffffff

    b = (b - c) & 0xffffffff
    b = (b - a) & 0xffffffff
    b = (b ^ (a << 10)) & 0xffffffff

    c = (c - a) & 0xffffffff
    c = (c - b) & 0xffffffff
    c = (c ^ (b >> 15)) & 0xffffffff
    return c, b, a


def jt_hash32(k, init_val):
    """
     hash() -- hash a variable-length key into a 32-bit value
     
     k : the key (the unaligned variable-length array of bytes)
     len : the length of the key, counting by bytes
     level : can be any 4-byte value
     Returns a 32-bit value. Every bit of the key affects every bit of
     the return value. Every 1-bit and 2-bit delta achieves avalanche.
     About 36+6len instructions.
     The best hash table sizes are powers of 2. There is no need to do
     mod a prime (mod is sooo slow!). If you need less than 32 bits,
     use a bitmask. For example, if you need only 10 bits, do
     h = (h & hashmask(10));
     In which case, the hash table should have hashsize(10) elements.
     If you are hashing n strings (JtUInt8 **)k, do it like this:
     for (i=0, h=0; i<n; ++i) h = hash( k[i], len[i], h);
    
     By Bob Jenkins, 1996. bob_jenkins@burtleburtle.net. You may use this
     code any way you wish, private, educational, or commercial. It's free.
    
     See http:burtleburtle.net/bob/  2010/02/12
     See http:burtleburtle.net/bob/hash/doobs.html  2010/02/12
    
     Use for hash table lookup, or anything where one collision in 2^32 is
     acceptable. Do NOT use for cryptographic purposes.
     
     This works on all machines. hash2() is identical to hash() on
     little-endian machines, except that the length has to be measured
     in ub4s instead of bytes. It is much faster than hash(). It
     requires
     -- that the key be an array of UInt32's, and
     -- that all your machines have the same endianness, and
     -- that the length be the number of UInt32's in the key
    """
    length = len(k)
    a = b = 0x9e3779b9
    c = init_val
    while length >= 12:
        a += k[0] + ((k[1] & 0xffffffff) << 8) + ((k[2] & 0xffffffff) << 16) + ((k[3] & 0xffffffff) << 24)
        a &= 0xffffffff
        b += k[4] + ((k[5] & 0xffffffff) << 8) + ((k[6] & 0xffffffff) << 16) + ((k[7] & 0xffffffff) << 24)
        b &= 0xffffffff
        c += k[8] + ((k[9] & 0xffffffff) << 8) + ((k[10] & 0xffffffff) << 16) + ((k[11] & 0xffffffff) << 24)
        c &= 0xffffffff
        a, b, c = mix(a, b, c)
        k = k[12:]
        length -= 12

    c += length
    if length == 11:
        c += k[10] << 24
        c += k[9] << 16
        c += k[8] << 8
        b += ((k[7] & 0xffffffff) << 24)
        b += ((k[6] & 0xffffffff) << 16)
        b += ((k[5] & 0xffffffff) << 8)
        b += k[4]
        a += ((k[3] & 0xffffffff) << 24)
        a += ((k[2] & 0xffffffff) << 16)
        a += ((k[1] & 0xffffffff) << 8)
        a += k[0]
    elif length == 10:
        c += k[9] << 16
        c += k[8] << 8
        b += ((k[7] & 0xffffffff) << 24)
        b += ((k[6] & 0xffffffff) << 16)
        b += ((k[5] & 0xffffffff) << 8)
        b += k[4]
        a += ((k[3] & 0xffffffff) << 24)
        a += ((k[2] & 0xffffffff) << 16)
        a += ((k[1] & 0xffffffff) << 8)
        a += k[0]
    elif length == 9:
        c += k[8] << 8
        b += ((k[7] & 0xffffffff) << 24)
        b += ((k[6] & 0xffffffff) << 16)
        b += ((k[5] & 0xffffffff) << 8)
        b += k[4]
        a += ((k[3] & 0xffffffff) << 24)
        a += ((k[2] & 0xffffffff) << 16)
        a += ((k[1] & 0xffffffff) << 8)
        a += k[0]
    elif length == 8:
        b += ((k[7] & 0xffffffff) << 24)
        b += ((k[6] & 0xffffffff) << 16)
        b += ((k[5] & 0xffffffff) << 8)
        b += k[4]
        a += ((k[3] & 0xffffffff) << 24)
        a += ((k[2] & 0xffffffff) << 16)
        a += ((k[1] & 0xffffffff) << 8)
        a += k[0]
    elif length == 7:
        b += ((k[6] & 0xffffffff) << 16)
        b += ((k[5] & 0xffffffff) << 8)
        b += k[4]
        a += ((k[3] & 0xffffffff) << 24)
        a += ((k[2] & 0xffffffff) << 16)
        a += ((k[1] & 0xffffffff) << 8)
        a += k[0]
    elif length == 6:
        b += ((k[5] & 0xffffffff) << 8)
        b += k[4]
        a += ((k[3] & 0xffffffff) << 24)
        a += ((k[2] & 0xffffffff) << 16)
        a += ((k[1] & 0xffffffff) << 8)
        a += k[0]
    elif length == 5:
        b += k[4]
        a += ((k[3] & 0xffffffff) << 24)
        a += ((k[2] & 0xffffffff) << 16)
        a += ((k[1] & 0xffffffff) << 8)
        a += k[0]
    elif length == 4:
        a += ((k[3] & 0xffffffff) << 24)
        a += ((k[2] & 0xffffffff) << 16)
        a += ((k[1] & 0xffffffff) << 8)
        a += k[0]
    elif length == 3:
        a += ((k[2] & 0xffffffff) << 16)
        a += ((k[1] & 0xffffffff) << 8)
        a += k[0]
    elif length == 2:
        a += ((k[1] & 0xffffffff) << 8)
        a += k[0]
    elif length == 1:
        a += k[0]

    a &= 0xffffffff
    b &= 0xffffffff
    c &= 0xffffffff
    a,b,c = mix(a, b, c)
    return c


def jt_hash16(k, init_val):
    if not k:
        return jt_hash32(b"", init_val)
    data = bytearray()
    for v in k:
        data += struct.pack("<H", v & 0xFFFF)
    return jt_hash32(bytes(data), init_val)


def jt_hash32_ints(k, init_val):
    if not k:
        return jt_hash32(b"", init_val)
    data = bytearray()
    for v in k:
        data += struct.pack("<I", v & 0xFFFFFFFF)
    return jt_hash32(bytes(data), init_val)
