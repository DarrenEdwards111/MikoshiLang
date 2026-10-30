The legacy `extended*.py` modules defined about 5,508 further names. 148 of them were reviewed during the audit:

* those with a real Wolfram meaning and a correct implementation were promoted or recorded under their area;
* 6 that the kernel already evaluates correctly (`AtomQ`, `ListQ`, `StringQ`, `ToString`, `ReplaceAll`,
  `Distance2D`) were recorded as kernel heads;
* `YoungTableaux` was briefly quarantined because the legacy version returned the size of a single tableau. It is now properly implemented and promoted (see above).

**Decision for the remaining names: keep them quarantined.** These are the other names on the list in
`mikoshilang/quarantined.json`. They raise
`NotImplementedError: <name> belongs to the quarantined, unreviewed library`.

Sampling during the audit showed why. The great majority are one-line rules that:

* hard-code a placeholder under a broad name. For example, `ANOVAOneWay[...]` always returns `1.0`,
  `AFTModel[...]` returns `{0}`, `ARCH[q]` returns the string `"ARCH(q)"`, and `AGCSignal[x]` is the identity;
* duplicate a reviewed head under a near-synonym.

Promoting any of them requires an actual implementation plus at least 3 differential cases, as with the 26
promoted heads. `tests/library/test_coverage.py::test_every_quarantined_name_raises` calls every remaining
quarantined name and fails if any returns a value.
