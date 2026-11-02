"""core.add_check: the Check: sentence for heads whose notes are built in code (CONVENTIONS §4)."""
import pytest

import mikoshilang  # noqa: F401
from mikoshilang.library import core
from mikoshilang.verification import stated


@pytest.fixture
def head():
    name = "AddCheckTestHead`"
    core.record(name, "test", "added", "a test head")
    yield name
    core.STATUS.pop(name, None)


def test_appends_sentence(head):
    core.add_check("reference (differential cases in tests/library).", head)
    notes = core.STATUS[head][2]
    assert notes == "a test head. Check: reference (differential cases in tests/library)."
    assert stated(notes)[0] == ["reference"]


def test_keeps_existing_punctuation(head):
    core.record(head, "test", "added", "ends with a full stop.")
    core.add_check("exact (exact input gives exact results), reference (differential cases in tests/library).", head)
    notes = core.STATUS[head][2]
    assert notes.startswith("ends with a full stop. Check: exact")
    assert stated(notes)[0] == ["exact", "reference"]


def test_already_stated_raises(head):
    core.add_check("reference (differential cases in tests/library).", head)
    with pytest.raises(ValueError, match="add_check: .* already states a check"):
        core.add_check("reference (differential cases in tests/library).", head)


def test_unregistered_raises():
    with pytest.raises(ValueError, match="add_check: NoSuchHeadXyz is not registered"):
        core.add_check("reference (differential cases in tests/library).", "NoSuchHeadXyz")

