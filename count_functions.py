#!/usr/bin/env python3
"""Report the enabled dispatcher; name counts are not correctness guarantees."""
def main():
    from mikoshilang.evaluate import _ensure_extended, _rules_by_head
    from mikoshilang.runtime import _quarantined
    _ensure_extended()
    print(f'Enabled extension heads: {len(_rules_by_head)} (not formally verified)')
    print(f'Quarantined legacy names: {len(_quarantined)} (core overloads may remain)')
    print('Language special forms are implemented separately in runtime.py.')
    for name in sorted(_rules_by_head):
        print(name)

if __name__ == '__main__':
    main()
