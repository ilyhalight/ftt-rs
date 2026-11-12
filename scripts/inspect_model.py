"""Dump the physical layout of a fastText .bin/.ftz model (format v12)."""

import struct
import sys
import tempfile
from pathlib import Path


def main(path: str) -> None:
    b = open(path, "rb").read()
    o = 0
    magic, ver = struct.unpack_from("<ii", b, o)
    o += 8
    print("magic", magic, "ver", ver)
    print("args", struct.unpack_from("<12i", b, o))
    o += 48
    print("t", struct.unpack_from("<d", b, o))
    o += 8
    print("dict header @", o, struct.unpack_from("<iiiqq", b, o))
    size, nw, nl, ntok, prune = struct.unpack_from("<iiiqq", b, o)
    o += 32
    for i in range(size):
        e = b.index(0, o)
        if i < 3 or i >= size - 2:
            print("entry", i, b[o:e][:40], "at", o)
        o = e + 1
        cnt, typ = struct.unpack_from("<qb", b, o)
        o += 9
        if i < 3 or i >= size - 2:
            print("   count/type", cnt, typ)
    print("after entries", o, "remaining", len(b) - o, "prune", prune)
    for i in range(prune):
        if i < 3:
            print("prune", i, struct.unpack_from("<ii", b, o))
        o += 8
    print("after prune", o, "bytes", b[o : o + 40].hex())
    print("quant", b[o])
    o += 1
    m, n = struct.unpack_from("<qq", b, o)
    print("matrix", m, n, "@", o)
    o += 16
    print("after hdr", o, "data bytes", m * n * 4, "remaining", len(b) - o)
    o += m * n * 4
    print("after data", o, "next", b[o : o + 48].hex())
    print("qout", b[o])
    o += 1
    print("out hdr", struct.unpack_from("<qq", b, o))
    o += 16
    m, n = struct.unpack_from("<qq", b, o - 16)
    o += m * n * 4
    print("after out data", o, "file", len(b), "remaining", len(b) - o)


if __name__ == "__main__":
    main(
        sys.argv[1]
        if len(sys.argv) > 1
        else str(Path(tempfile.gettempdir()) / "ftmodels" / "lid.176.ftz")
    )
