"""Values of built-in symbols that evaluate to an expression, registered by library modules.

``SYMBOL_VALUES[name]`` is what the bare symbol evaluates to when the user has not assigned it (an OwnValue
set by the user always wins). Example: the Wolfram batch W14 registers ``Red -> RGBColor[1, 0, 0]`` and
``Thick -> Thickness[Large]``. Values must be fully evaluated expressions (they are returned as they are).
"""
SYMBOL_VALUES = {}
