#!/usr/bin/env python3
"""Report the audited bundled library: every exposed head with its audit status.

Statuses: verified (legacy behaviour confirmed against references), fixed (wrong legacy behaviour
corrected), promoted (formerly quarantined, now implemented and tested), kernel (implemented by the
evaluator kernel), error (raises a clear NotImplementedError pointing at the Wolfram replacement),
quarantined (raises NotImplementedError). See docs/library_audit.md.
"""
from collections import Counter


def main():
    from mikoshilang import library
    from mikoshilang.runtime import _quarantined
    counts = Counter(s for _area, s, _notes in library.STATUS.values())
    print(f"Audited library heads: {len(library.STATUS)}")
    for status in ("verified", "fixed", "promoted", "kernel", "error", "quarantined"):
        print(f"  {status:12s} {counts.get(status, 0)}")
    print(f"Quarantined legacy names (raise NotImplementedError): {len(_quarantined)}")
    for name in sorted(library.STATUS):
        area, status, _notes = library.STATUS[name]
        print(f"{name}\t{area}\t{status}")


if __name__ == "__main__":
    main()
