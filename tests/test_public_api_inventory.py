"""Keep the bounded public-API audit complete as the exported surface evolves.

This is an inventory/evidence guard, not proof that the tested semantics are
complete. Individual test suites carry the value and property assertions.
"""
from __future__ import annotations

import ast
from collections import Counter
import inspect
import json
from pathlib import Path

import pytest
import mikoshilang as ml

ROOT = Path(__file__).resolve().parent


def _manifest():
    return json.loads((ROOT / 'public_api_audit.json').read_text(encoding='utf-8'))['exports']


def _validate_surface(rows):
    names = [row['name'] for row in rows]
    assert len(names) == len(set(names)), 'Duplicate audit records'
    assert set(names) == set(ml.__all__), {
        'missing': sorted(set(ml.__all__) - set(names)),
        'stale': sorted(set(names) - set(ml.__all__)),
    }


def _declarations(filename):
    tree = ast.parse((ROOT / filename).read_text(encoding='utf-8'))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'COVERED_EXPORTS' for t in node.targets):
            return set(ast.literal_eval(node.value)), tree
    raise AssertionError(f'{filename}: missing literal COVERED_EXPORTS declaration')


def test_audit_manifest_covers_current_public_surface_exactly():
    _validate_surface(_manifest())
    assert len(ml.__all__) == len(set(ml.__all__)), 'Duplicate public exports'


def test_batch_declarations_partition_public_exports():
    rows = _manifest()
    counts = Counter()
    for filename in sorted({row['test_file'] for row in rows}):
        assert Path(filename).name == filename and filename.startswith('test_public_api_')
        declared, _ = _declarations(filename)
        expected = {row['name'] for row in rows if row['test_file'] == filename}
        assert declared == expected, filename
        counts.update(declared)
    assert set(counts) == set(ml.__all__)
    assert all(count == 1 for count in counts.values()), 'Overlapping or duplicated audit batches'


def test_every_audit_record_references_real_tests_and_states_limitations():
    rows = _manifest()
    functions = {}
    for filename in {row['test_file'] for row in rows}:
        _, tree = _declarations(filename)
        functions[filename] = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith('test_')}
    for row in rows:
        assert row['status'] == 'checked', row['name']
        assert row['tests'] and set(row['tests']) <= functions[row['test_file']], row['name']
        assert isinstance(row['limitations'], list), row['name']
        assert isinstance(row['findings'], list), row['name']
        obj = getattr(ml, row['name'])
        kind = 'class' if inspect.isclass(obj) else 'function' if inspect.isfunction(obj) else 'callable' if callable(obj) else 'constant'
        assert row['kind'] == kind, row['name']


def test_inventory_guard_rejects_missing_export_records():
    rows = _manifest()
    assert rows
    with pytest.raises(AssertionError):
        _validate_surface(rows[:-1])


def test_inventory_guard_rejects_duplicate_export_records():
    rows = _manifest()
    with pytest.raises(AssertionError, match='Duplicate'):
        _validate_surface([*rows, rows[0]])
