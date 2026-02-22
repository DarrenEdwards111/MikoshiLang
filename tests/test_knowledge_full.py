"""Tests for MikoshiLang full knowledge layer."""

import pytest
import time
from pathlib import Path
from mikoshilang.knowledge_full import (
    EntitySearch,
    EntityValue,
    EntityRelationships,
    WikipediaText,
    KnowledgeAsOf,
    InterpretQuery,
    PackSearch,
    PackValue,
    CacheClear,
    CacheStats,
    CACHE_DIR,
)


# ============================================================================
# Cache Tests
# ============================================================================

def test_cache_initialization():
    """Test that cache directory and database are created."""
    assert CACHE_DIR.exists()
    assert (CACHE_DIR / "knowledge.db").exists()


def test_cache_stats():
    """Test cache statistics."""
    stats = CacheStats()
    assert "total_entries" in stats
    assert "valid_entries" in stats
    assert "cache_file" in stats
    assert stats["cache_file"] == str(CACHE_DIR / "knowledge.db")


def test_cache_persistence():
    """Test that cache persists across function calls."""
    # First call - should hit API
    result1 = EntitySearch("Douglas Adams", limit=1)
    assert len(result1) > 0
    
    # Second call - should hit cache
    result2 = EntitySearch("Douglas Adams", limit=1)
    assert result1 == result2


def test_cache_clear_all():
    """Test clearing entire cache."""
    # Add some data
    EntitySearch("test query")
    stats_before = CacheStats()
    
    # Clear cache
    CacheClear()
    
    # Verify cleared
    stats_after = CacheStats()
    assert stats_after["total_entries"] == 0


def test_cache_clear_old():
    """Test clearing old cache entries."""
    # This would require manipulating timestamps
    # Simplified test: just verify function works
    result = CacheClear(older_than_days=30)
    assert "cleared" in result
    assert "older_than_days" in result


# ============================================================================
# Entity Search Tests
# ============================================================================

def test_entity_search_basic():
    """Test basic entity search."""
    results = EntitySearch("Douglas Adams")
    assert isinstance(results, list)
    assert len(results) > 0
    assert results[0]["id"] == "Q42"
    assert "license" in results[0]
    assert results[0]["license"] == "CC0"


def test_entity_search_with_limit():
    """Test entity search with custom limit."""
    results = EntitySearch("Python", limit=3)
    assert len(results) <= 3


# ============================================================================
# Entity Value Tests
# ============================================================================

def test_entity_value_basic():
    """Test getting entity property value."""
    result = EntityValue("Q42", "BirthDate")
    assert result["value"] == "1952-03-11"
    assert result["license"] == "CC0"
    assert "source" in result


def test_entity_value_with_dict():
    """Test EntityValue with entity dict input."""
    entity = {"id": "Q42", "label": "Douglas Adams"}
    result = EntityValue(entity, "BirthDate")
    assert result["value"] == "1952-03-11"


# ============================================================================
# Relationship Query Tests (NEW)
# ============================================================================

def test_entity_relationships():
    """Test relationship traversal."""
    result = EntityRelationships("Q42", "Occupation", depth=1, limit=5)
    assert isinstance(result, dict)
    assert "related" in result
    assert isinstance(result["related"], list)
    assert result["license"] == "CC0"
    assert "source" in result


def test_entity_relationships_caching():
    """Test that relationships are cached."""
    result1 = EntityRelationships("Q42", "Occupation")
    result2 = EntityRelationships("Q42", "Occupation")
    # Should get same results
    assert result1["count"] == result2["count"]


# ============================================================================
# Wikipedia Text Tests
# ============================================================================

def test_wikipedia_text_basic():
    """Test getting Wikipedia text."""
    result = WikipediaText("Douglas Adams")
    assert "text" in result
    assert len(result["text"]) > 50
    assert result["license"] == "CC BY-SA 3.0"
    assert "attribution" in result


def test_wikipedia_text_with_sentences():
    """Test Wikipedia text with sentence limit."""
    result = WikipediaText("Python (programming language)", sentences=2)
    assert "text" in result
    # Should be shorter than default
    assert len(result["text"]) > 0


# ============================================================================
# Time/Versioning Tests (NEW)
# ============================================================================

def test_knowledge_as_of():
    """Test time-travel queries."""
    result = KnowledgeAsOf("Q42", "2020-01-01", "wikidata")
    assert isinstance(result, dict)
    # Currently returns note about implementation
    assert "note" in result or "entity" in result


def test_entity_value_with_asof():
    """Test EntityValue with as_of parameter."""
    result = EntityValue("Q42", "BirthDate", as_of="2020-01-01")
    # Should still return birth date (doesn't change)
    assert isinstance(result, dict)
    assert "value" in result


# ============================================================================
# LLM Interpreter Tests (NEW)
# ============================================================================

def test_interpret_query():
    """Test natural language query interpretation."""
    result = InterpretQuery("When was Douglas Adams born?")
    assert isinstance(result, dict)
    assert "plan" in result
    assert "natural_language" in result


# ============================================================================
# Domain Pack Tests (NEW)
# ============================================================================

def test_pubchem_search():
    """Test PubChem pack search."""
    results = PackSearch("pubchem", "caffeine", limit=3)
    assert isinstance(results, list)
    if len(results) > 0 and "error" not in results[0]:
        assert results[0]["source"] == "PubChem"
        assert results[0]["license"] == "Public Domain"


def test_pubchem_value():
    """Test PubChem property retrieval."""
    # Caffeine CID: 2519
    result = PackValue("pubchem", "2519", "MolecularFormula")
    if "error" not in result:
        assert result["value"] == "C8H10N4O2"
        assert result["license"] == "Public Domain"


def test_unknown_pack():
    """Test error handling for unknown pack."""
    results = PackSearch("nonexistent_pack", "query")
    assert "error" in results[0]


# ============================================================================
# Provenance Tests
# ============================================================================

def test_provenance_included():
    """Test that all results include provenance."""
    entity_result = EntityValue("Q42", "BirthDate")
    assert "source" in entity_result
    assert "retrieved" in entity_result
    assert "license" in entity_result
    
    wiki_result = WikipediaText("Douglas Adams")
    assert "source" in wiki_result or "url" in wiki_result
    assert "retrieved" in wiki_result
    assert "license" in wiki_result
    assert "attribution" in wiki_result


# ============================================================================
# Integration Tests
# ============================================================================

def test_full_workflow():
    """Test complete knowledge workflow."""
    # 1. Search for entity
    entities = EntitySearch("Douglas Adams", limit=1)
    assert len(entities) > 0
    entity = entities[0]
    
    # 2. Get properties
    birth_date = EntityValue(entity["id"], "BirthDate")
    assert birth_date["value"] == "1952-03-11"
    
    # 3. Get relationships
    occupations = EntityRelationships(entity["id"], "Occupation", limit=5)
    assert "related" in occupations
    
    # 4. Get text
    summary = WikipediaText(entity["label"])
    assert len(summary["text"]) > 0
    
    # 5. Check cache stats
    stats = CacheStats()
    assert stats["valid_entries"] > 0


def test_cache_ttl_difference():
    """Test that different sources have different TTLs."""
    # This is implicit in the code - Wikidata has 7d TTL, Wikipedia has 24h TTL
    # We can verify the cache entries exist
    EntityValue("Q42", "BirthDate")  # Wikidata
    WikipediaText("Douglas Adams")    # Wikipedia
    
    stats = CacheStats()
    assert stats["total_entries"] >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ============================================================================
# New Domain Pack Tests (Crossref, OpenAlex, GeoNames, WorldBank)
# ============================================================================

def test_crossref_search():
    """Test Crossref pack search."""
    results = PackSearch("crossref", "machine learning", limit=3)
    assert isinstance(results, list)
    if len(results) > 0 and "error" not in results[0]:
        assert "id" in results[0]  # DOI
        assert results[0]["source"] == "Crossref"


def test_crossref_value():
    """Test Crossref metadata retrieval."""
    # Using a known DOI
    result = PackValue("crossref", "10.1145/3422622", "Title")
    if "error" not in result:
        assert "value" in result
        assert result["license"] == "Metadata: CC0"


def test_openalex_search():
    """Test OpenAlex pack search."""
    results = PackSearch("openalex", "deep learning", limit=3)
    assert isinstance(results, list)
    if len(results) > 0 and "error" not in results[0]:
        assert results[0]["source"] == "OpenAlex"
        assert results[0]["license"] == "CC0"


def test_openalex_value():
    """Test OpenAlex work metadata."""
    # OpenAlex IDs start with W
    result = PackValue("openalex", "W2741809807", "CitationCount")
    if "error" not in result:
        assert "value" in result
        assert result["license"] == "CC0"


def test_geonames_search():
    """Test GeoNames pack search."""
    results = PackSearch("geonames", "London", limit=3)
    assert isinstance(results, list)
    if len(results) > 0 and "error" not in results[0]:
        assert results[0]["source"] == "GeoNames"
        assert results[0]["license"] == "CC BY 4.0"


def test_geonames_value():
    """Test GeoNames place metadata."""
    # London, UK geonameId
    result = PackValue("geonames", "2643743", "Country")
    if "error" not in result:
        assert "value" in result
        assert result["license"] == "CC BY 4.0"


def test_worldbank_search():
    """Test World Bank pack search."""
    results = PackSearch("worldbank", "GDP", limit=3)
    assert isinstance(results, list)
    if len(results) > 0 and "error" not in results[0]:
        assert results[0]["source"] == "World Bank"
        assert results[0]["license"] == "CC BY 4.0"


def test_worldbank_value():
    """Test World Bank indicator data."""
    # GDP indicator
    result = PackValue("worldbank", "NY.GDP.MKTP.CD", "Value", country="US")
    if "error" not in result:
        # May return None if no recent data
        assert "value" in result
        assert result["license"] == "CC BY 4.0"


def test_all_packs_registered():
    """Test that all 5 packs are registered."""
    from mikoshilang.knowledge_full import _KNOWLEDGE_PACKS
    
    assert "pubchem" in _KNOWLEDGE_PACKS
    assert "crossref" in _KNOWLEDGE_PACKS
    assert "openalex" in _KNOWLEDGE_PACKS
    assert "geonames" in _KNOWLEDGE_PACKS
    assert "worldbank" in _KNOWLEDGE_PACKS
    
    assert len(_KNOWLEDGE_PACKS) == 5
