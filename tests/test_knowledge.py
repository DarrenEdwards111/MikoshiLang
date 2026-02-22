"""Tests for MikoshiLang knowledge layer."""

import pytest
from mikoshilang import parse_and_eval


def test_entity_search():
    """Test entity search"""
    result = parse_and_eval('EntitySearch["Douglas Adams"]')
    assert isinstance(result, list)
    assert len(result) > 0
    assert result[0]["id"] == "Q42"  # Douglas Adams
    assert "label" in result[0]
    assert "description" in result[0]


def test_entity_search_with_type():
    """Test entity search with type hint"""
    result = parse_and_eval('EntitySearch["Douglas Adams", "Person"]')
    assert isinstance(result, list)
    assert len(result) > 0
    # Should find Douglas Adams with the Person hint
    assert result[0]["id"] == "Q42"


def test_entity_value_birth_date():
    """Test fetching birth date"""
    result = parse_and_eval('EntityValue["Q42", "BirthDate"]')
    assert isinstance(result, dict)
    assert result["value"] == "1952-03-11"
    assert result["property"] == "BirthDate"
    assert result["pid"] == "P569"


def test_entity_value_occupation():
    """Test fetching occupation (multi-value)"""
    result = parse_and_eval('EntityValue["Q42", "Occupation"]')
    assert isinstance(result, dict)
    assert "value" in result
    # Should be list of entities (writer, etc.)
    assert isinstance(result["value"], list)


def test_entity_properties():
    """Test fetching all properties"""
    result = parse_and_eval('EntityProperties["Q42"]')
    assert isinstance(result, dict)
    assert "properties" in result
    properties = result["properties"]
    assert "BirthDate" in properties
    assert properties["BirthDate"] == "1952-03-11"


def test_wikipedia_summary():
    """Test Wikipedia summary"""
    result = parse_and_eval('WikipediaSummary["Douglas Adams"]')
    assert isinstance(result, dict)
    assert "summary" in result
    assert len(result["summary"]) > 50
    assert "Douglas" in result["summary"]
    assert result["license"] == "CC BY-SA 3.0 (Wikipedia)"


def test_wikipedia_summary_with_sentences():
    """Test Wikipedia summary with sentence limit"""
    result = parse_and_eval('WikipediaSummary["Python (programming language)", 2]')
    assert isinstance(result, dict)
    assert "summary" in result
    # Should be roughly 2 sentences (not exact due to splitting logic)
    assert len(result["summary"]) > 0


def test_disambiguate():
    """Test disambiguation"""
    result = parse_and_eval('Disambiguate["Moon"]')
    assert isinstance(result, dict)
    assert "candidates" in result
    assert len(result["candidates"]) > 1
    # Should have both celestial body and film
    descriptions = [c.get("description", "") for c in result["candidates"]]
    assert any("natural satellite" in d.lower() for d in descriptions)


def test_disambiguate_with_context():
    """Test disambiguation with context"""
    result = parse_and_eval('Disambiguate["Adams", "writer"]')
    assert isinstance(result, dict)
    assert "candidates" in result
    # Should return multiple Adams-related results
    assert len(result["candidates"]) > 0


def test_entity_convenience():
    """Test Entity convenience function"""
    result = parse_and_eval('Entity["Douglas Adams"]')
    assert isinstance(result, dict)
    assert result["id"] == "Q42"


def test_entity_with_type():
    """Test Entity with type hint"""
    result = parse_and_eval('Entity["Writer", "Douglas Adams"]')
    assert isinstance(result, dict)
    # Should find Douglas Adams (first result with Writer hint)
    assert "id" in result
    assert result["id"] == "Q42"


def test_invalid_entity():
    """Test error handling for non-existent entity"""
    result = parse_and_eval('EntityValue["Q999999999", "BirthDate"]')
    assert isinstance(result, dict)
    # Should return error or empty value
    assert "error" in result or result.get("value") is None


def test_invalid_property():
    """Test error handling for non-existent property"""
    result = parse_and_eval('EntityValue["Q42", "NonExistentProperty"]')
    assert isinstance(result, dict)
    assert "error" in result


def test_provenance_tracking():
    """Test that provenance (source + timestamp) is included"""
    result = parse_and_eval('EntityValue["Q42", "BirthDate"]')
    assert "source" in result
    assert "retrieved" in result
    assert result["source"].startswith("https://www.wikidata.org")


def test_license_attribution():
    """Test license information is present"""
    wikidata_result = parse_and_eval('EntityValue["Q42", "BirthDate"]')
    assert wikidata_result.get("license") == "CC0 (Wikidata)"
    
    wikipedia_result = parse_and_eval('WikipediaSummary["Douglas Adams"]')
    assert wikipedia_result.get("license") == "CC BY-SA 3.0 (Wikipedia)"
    assert "attribution" in wikipedia_result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
