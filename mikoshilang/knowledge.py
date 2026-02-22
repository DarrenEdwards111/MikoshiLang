"""
MikoshiLang Knowledge Layer - Entity Graph + Text Retrieval

Architecture:
1. Entity layer (Wikidata) - structured facts, QIDs, properties
2. Text layer (Wikipedia) - summaries, context, citations
3. Deterministic tools - no LLM hallucination
4. Provenance tracking - sources + timestamps

License compliance:
- Wikidata: CC0 (public domain)
- Wikipedia: CC BY-SA 3.0 (attribution required)
"""

import requests
import json
import time
from datetime import datetime, timezone
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Any

# Wikidata API endpoints
WIKIDATA_API = "https://www.wikidata.org/w/api.php"
WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"

# User-Agent header (required by Wikimedia APIs)
HEADERS = {
    "User-Agent": "MikoshiLang/1.0 (https://github.com/DarrenEdwards111/MikoshiLang; mikoshi.ai) Python/3.x"
}

# Canonical property mappings (Mikoshi name → Wikidata PID)
CANONICAL_PROPERTIES = {
    "BirthDate": "P569",
    "DeathDate": "P570",
    "BirthPlace": "P19",
    "Occupation": "P106",
    "InstanceOf": "P31",
    "SubclassOf": "P279",
    "Country": "P17",
    "Population": "P1082",
    "Coordinate": "P625",
    "Mass": "P2067",
    "Height": "P2048",
    "Director": "P57",
    "Author": "P50",
    "Publisher": "P123",
    "PublicationDate": "P577",
    "ISBN": "P212",
    "DOI": "P356",
    "ChemicalFormula": "P274",
    "AtomicNumber": "P1086",
    "MolarMass": "P2067",
}

# Reverse mapping (PID → human-readable name)
PID_TO_NAME = {v: k for k, v in CANONICAL_PROPERTIES.items()}

# Cache for entity lookups (keyed by query + type)
@lru_cache(maxsize=1000)
def _entity_search_cached(query: str, entity_type: Optional[str], limit: int) -> str:
    """Cached Wikidata entity search"""
    # If entity_type provided, append it to the search query as a hint
    search_query = f"{query} {entity_type}" if entity_type else query
    
    params = {
        "action": "wbsearchentities",
        "format": "json",
        "language": "en",
        "search": search_query,
        "limit": limit,
    }
    
    try:
        response = requests.get(WIKIDATA_API, params=params, headers=HEADERS, timeout=10)
        response.raise_for_status()
        return response.text
    except Exception as e:
        return json.dumps({"error": str(e)})


def EntitySearch(query: str, entity_type: Optional[str] = None, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Search for entities in Wikidata.
    
    Args:
        query: Search string (e.g., "Douglas Adams", "Moon")
        entity_type: Optional type hint ("Person", "Film", "Place", etc.)
        limit: Max results to return
    
    Returns:
        List of candidate entities with:
        - id: Wikidata QID (e.g., "Q42")
        - label: Primary name
        - description: Short description
        - url: Wikidata URL
        - retrieved: Timestamp
    
    Example:
        >>> EntitySearch["Douglas Adams", "Person"]
        [{"id": "Q42", "label": "Douglas Adams", 
          "description": "English writer and humorist", ...}]
    """
    response_text = _entity_search_cached(query, entity_type, limit)
    data = json.loads(response_text)
    
    if "error" in data:
        return [{"error": data["error"]}]
    
    results = []
    for item in data.get("search", []):
        results.append({
            "id": item.get("id"),
            "label": item.get("label"),
            "description": item.get("description", ""),
            "url": f"https://www.wikidata.org/wiki/{item.get('id')}",
            "retrieved": datetime.now(timezone.utc).isoformat() + "Z",
            "source": "Wikidata (CC0)",
        })
    
    return results


@lru_cache(maxsize=500)
def _entity_data_cached(qid: str) -> str:
    """Cached Wikidata entity data fetch"""
    params = {
        "action": "wbgetentities",
        "format": "json",
        "ids": qid,
        "languages": "en",
    }
    
    try:
        response = requests.get(WIKIDATA_API, params=params, headers=HEADERS, timeout=10)
        response.raise_for_status()
        return response.text
    except Exception as e:
        return json.dumps({"error": str(e)})


def EntityValue(entity_id: str, property_name: str) -> Dict[str, Any]:
    """
    Get a specific property value for an entity.
    
    Args:
        entity_id: Wikidata QID (e.g., "Q42") or entity dict from EntitySearch
        property_name: Canonical property name (e.g., "BirthDate", "Occupation")
                      or Wikidata PID (e.g., "P569")
    
    Returns:
        Dict with:
        - value: The property value(s)
        - property: Property name
        - pid: Wikidata property ID
        - source: Source URL
        - retrieved: Timestamp
    
    Example:
        >>> EntityValue["Q42", "BirthDate"]
        {"value": "1952-03-11", "property": "BirthDate", "pid": "P569", ...}
    """
    # Handle entity dict input
    if isinstance(entity_id, dict):
        entity_id = entity_id.get("id")
    
    # Resolve property name to PID
    if property_name.startswith("P") and property_name[1:].isdigit():
        pid = property_name
        prop_name = PID_TO_NAME.get(pid, pid)
    else:
        pid = CANONICAL_PROPERTIES.get(property_name, property_name)
        prop_name = property_name
    
    # Fetch entity data
    response_text = _entity_data_cached(entity_id)
    data = json.loads(response_text)
    
    if "error" in data:
        return {"error": data["error"]}
    
    entity_data = data.get("entities", {}).get(entity_id, {})
    claims = entity_data.get("claims", {}).get(pid, [])
    
    if not claims:
        return {
            "value": None,
            "property": prop_name,
            "pid": pid,
            "entity": entity_id,
            "error": f"Property {prop_name} ({pid}) not found",
        }
    
    # Extract values
    values = []
    for claim in claims:
        mainsnak = claim.get("mainsnak", {})
        datavalue = mainsnak.get("datavalue", {})
        
        if datavalue.get("type") == "time":
            # Time values (dates)
            time_val = datavalue.get("value", {}).get("time", "")
            # Strip leading + and precision info
            time_val = time_val.lstrip("+").split("T")[0]
            values.append(time_val)
        elif datavalue.get("type") == "wikibase-entityid":
            # Entity references (linked items)
            linked_id = datavalue.get("value", {}).get("id")
            values.append({"id": linked_id, "url": f"https://www.wikidata.org/wiki/{linked_id}"})
        elif datavalue.get("type") == "string":
            values.append(datavalue.get("value"))
        elif datavalue.get("type") == "quantity":
            amount = datavalue.get("value", {}).get("amount")
            unit = datavalue.get("value", {}).get("unit", "")
            values.append({"amount": amount, "unit": unit})
        elif datavalue.get("type") == "globecoordinate":
            coord = datavalue.get("value", {})
            values.append({
                "latitude": coord.get("latitude"),
                "longitude": coord.get("longitude"),
            })
        else:
            values.append(str(datavalue.get("value")))
    
    # Return single value if only one, else list
    final_value = values[0] if len(values) == 1 else values
    
    return {
        "value": final_value,
        "property": prop_name,
        "pid": pid,
        "entity": entity_id,
        "source": f"https://www.wikidata.org/wiki/{entity_id}",
        "retrieved": datetime.now(timezone.utc).isoformat() + "Z",
        "license": "CC0 (Wikidata)",
    }


def EntityProperties(entity_id: str) -> Dict[str, Any]:
    """
    Get all available properties for an entity.
    
    Args:
        entity_id: Wikidata QID or entity dict
    
    Returns:
        Dict mapping property names to values
    
    Example:
        >>> EntityProperties["Q42"]
        {"BirthDate": "1952-03-11", "Occupation": [...], ...}
    """
    if isinstance(entity_id, dict):
        entity_id = entity_id.get("id")
    
    response_text = _entity_data_cached(entity_id)
    data = json.loads(response_text)
    
    if "error" in data:
        return {"error": data["error"]}
    
    entity_data = data.get("entities", {}).get(entity_id, {})
    claims = entity_data.get("claims", {})
    
    properties = {}
    for pid, claim_list in claims.items():
        # Use canonical name if available, else PID
        prop_name = PID_TO_NAME.get(pid, pid)
        result = EntityValue(entity_id, pid)
        if "error" not in result:
            properties[prop_name] = result["value"]
    
    return {
        "entity": entity_id,
        "properties": properties,
        "source": f"https://www.wikidata.org/wiki/{entity_id}",
        "retrieved": datetime.now(timezone.utc).isoformat() + "Z",
    }


@lru_cache(maxsize=500)
def _wikipedia_summary_cached(title: str) -> str:
    """Cached Wikipedia summary fetch"""
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts",
        "exintro": True,
        "explaintext": True,
    }
    
    try:
        response = requests.get(WIKIPEDIA_API, params=params, headers=HEADERS, timeout=10)
        response.raise_for_status()
        return response.text
    except Exception as e:
        return json.dumps({"error": str(e)})


def WikipediaSummary(title: str, sentences: int = 3) -> Dict[str, Any]:
    """
    Get Wikipedia article summary (lead section).
    
    Args:
        title: Wikipedia article title
        sentences: Number of sentences to return (approximate)
    
    Returns:
        Dict with:
        - summary: Text summary
        - title: Article title
        - url: Wikipedia URL
        - retrieved: Timestamp
        - license: CC BY-SA 3.0
    
    Example:
        >>> WikipediaSummary["Douglas Adams"]
        {"summary": "Douglas Noel Adams was an English author...", ...}
    """
    response_text = _wikipedia_summary_cached(title)
    data = json.loads(response_text)
    
    if "error" in data:
        return {"error": data["error"]}
    
    pages = data.get("query", {}).get("pages", {})
    page = next(iter(pages.values()), {})
    
    if "missing" in page:
        return {"error": f"Wikipedia article '{title}' not found"}
    
    extract = page.get("extract", "")
    
    # Truncate to requested sentences (rough)
    sentences_list = extract.split(". ")[:sentences]
    summary = ". ".join(sentences_list)
    if not summary.endswith("."):
        summary += "."
    
    return {
        "summary": summary,
        "title": page.get("title"),
        "url": f"https://en.wikipedia.org/wiki/{page.get('title', '').replace(' ', '_')}",
        "retrieved": datetime.now(timezone.utc).isoformat() + "Z",
        "license": "CC BY-SA 3.0 (Wikipedia)",
        "attribution": "Source: Wikipedia contributors",
    }


def Disambiguate(query: str, context: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Find disambiguation candidates for ambiguous queries.
    
    Args:
        query: Ambiguous term (e.g., "Moon")
        context: Optional context hint (e.g., "film", "astronomy")
    
    Returns:
        List of candidates with descriptions for user to choose
    
    Example:
        >>> Disambiguate["Moon", "film"]
        [{"id": "Q219562", "label": "Moon", "description": "2009 film by Duncan Jones"}, ...]
    """
    # If context provided, use it in search
    search_query = f"{query} {context}" if context else query
    
    # Search Wikidata
    entities = EntitySearch(search_query, limit=10)
    
    # Also check for Wikipedia disambiguation page
    wiki_params = {
        "action": "query",
        "format": "json",
        "titles": f"{query} (disambiguation)",
        "prop": "revisions",
        "rvprop": "content",
        "rvslots": "main",
    }
    
    try:
        wiki_response = requests.get(WIKIPEDIA_API, params=wiki_params, timeout=10)
        wiki_data = wiki_response.json()
        wiki_pages = wiki_data.get("query", {}).get("pages", {})
        
        # If disambiguation page exists, note it
        has_disambig = any("disambiguation" in str(p.get("title", "")).lower() 
                          for p in wiki_pages.values())
    except:
        has_disambig = False
    
    return {
        "query": query,
        "context": context,
        "candidates": entities,
        "has_disambiguation_page": has_disambig,
        "note": "Multiple entities found - please refine query or select by ID",
    }


# Convenience function for natural language queries
def Entity(name_or_type: str, name: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience wrapper for entity lookup.
    
    Usage:
        Entity["Douglas Adams"]           # Search by name
        Entity["Person", "Douglas Adams"] # Search with type hint
    
    Returns the top entity match or disambiguation prompt.
    """
    if name is None:
        # Single argument: just a name
        results = EntitySearch(name_or_type, limit=5)
    else:
        # Two arguments: type, name
        results = EntitySearch(name, entity_type=name_or_type, limit=5)
    
    if not results:
        return {"error": "No entities found"}
    
    # Always return the first result if we have results
    return results[0]


# Export all knowledge functions
__all__ = [
    "EntitySearch",
    "EntityValue",
    "EntityProperties",
    "WikipediaSummary",
    "Disambiguate",
    "Entity",
    "CANONICAL_PROPERTIES",
]
