## MikoshiLang Full Knowledge Layer

**Production-grade entity framework** with all 8 components from the blueprint.

## Architecture Overview

```
User Query
    ↓
LLM Interpreter (optional) → Structured Query Plan
    ↓
Knowledge API (5 core tools)
    ├─ Entity Search      → Wikidata
    ├─ Entity Facts       → Wikidata + Properties
    ├─ Relationships      → Wikidata SPARQL (graph traversal)
    ├─ Text Retrieval     → Wikipedia
    └─ Time/Versioning    → Historical snapshots
    ↓
Domain Packs (extensible)
    ├─ PubChem           → Chemistry
    ├─ Crossref          → Papers (coming soon)
    ├─ OpenAlex          → Scholarly graph (coming soon)
    └─ Custom packs      → Your domain
    ↓
Persistent Cache (SQLite)
    ├─ TTL by source (7d Wikidata, 24h Wikipedia)
    ├─ Provenance tracking
    └─ License metadata
    ↓
Results + Citations
```

## Components

### 1. Entity Graph (Wikidata)

Full entity framework with QIDs, PIDs, properties, and relationships.

**50+ canonical properties:**
- People: BirthDate, DeathDate, Occupation, Spouse, Child, etc.
- Places: Country, Capital, Population, Coordinate, etc.
- Things: Mass, Height, ChemicalFormula, AtomicNumber, etc.
- Works: Author, Director, Publisher, DOI, ISBN, etc.
- Organizations: Founded, CEO, Headquarters, Industry, etc.

### 2. Text Layer (Wikipedia)

Summaries, context, citations with full attribution.

**Features:**
- Section extraction
- Sentence limiting
- Revision history (as-of queries)
- CC BY-SA 3.0 compliance

### 3. Five Core Tools

#### Tool 1: Entity Search

```python
EntitySearch["Douglas Adams"]
# Returns: [{"id": "Q42", "label": "Douglas Adams", ...}]

EntitySearch["Moon", "Film", 10]  # With type hint and limit
```

**Features:**
- Persistent caching (7d TTL)
- Provenance tracking
- Type hints for disambiguation

#### Tool 2: Entity Facts

```python
EntityValue["Q42", "BirthDate"]
# Returns: {"value": "1952-03-11", "source": "...", "license": "CC0"}

EntityValue["Q42", "Occupation"]  # Multi-value properties
EntityValue["Q42", "BirthDate", "2020-01-01"]  # Time-travel (as-of)
```

**Features:**
- 50+ canonical properties
- Automatic value type handling (dates, entities, quantities, coordinates)
- Historical snapshots (as-of parameter)
- Persistent caching

#### Tool 3: Relationship Queries ⭐ NEW

```python
EntityRelationships["Q42", "Occupation"]
# Returns: {"related": [{"id": "Q36180", "label": "writer", ...}], ...}

EntityRelationships["Q42", "Influenced", 2, 10]  # depth=2, limit=10
# Multi-hop graph traversal
```

**Features:**
- SPARQL-powered graph queries
- Configurable depth (1-3 hops)
- Result limiting
- Cached for performance

#### Tool 4: Text Retrieval

```python
WikipediaText["Douglas Adams"]
# Returns: {"text": "Douglas Noel Adams was...", "license": "CC BY-SA 3.0"}

WikipediaText["Python", "History", 5]  # Section + sentence limit
WikipediaText["Python", as_of="2020-01-01"]  # Historical version
```

**Features:**
- Lead section or specific sections
- Sentence limiting
- Revision history
- Full attribution

#### Tool 5: Time/Versioning ⭐ NEW

```python
KnowledgeAsOf["Q42", "2020-01-01", "wikidata"]
# Returns snapshot as of Jan 1, 2020

EntityValue["Q42", "Occupation", as_of="2015-06-01"]
# Property value as of specific date
```

**Features:**
- Revision API integration
- Temporal queries
- Historical fact-checking

### 4. LLM Interpreter Layer ⭐ NEW

```python
InterpretQuery["When was Douglas Adams born?"]
# Returns:
# {
#   "plan": [
#     {"step": 1, "action": "EntitySearch", "args": ["Douglas Adams"]},
#     {"step": 2, "action": "EntityValue", "args": ["Q42", "BirthDate"]}
#   ],
#   "execution": {...},
#   "answer": "1952-03-11"
# }
```

**Features:**
- Natural language → structured query
- Multi-step planning
- Automatic execution
- Works with Anthropic Claude / OpenAI / Ollama

### 5. Persistent Caching + Provenance

**Cache Database:** `~/.mikoshilang/knowledge_cache/knowledge.db`

**Features:**
- SQLite backend (lightweight, portable)
- TTL by source:
  - Wikidata entities: 7 days
  - Wikipedia text: 24 hours
  - Relationships: 7 days
- Automatic expiry cleanup
- Provenance tracking:
  - Source URL
  - Retrieved timestamp
  - License info

**Cache Management:**

```python
CacheStats[]
# Returns: {"total_entries": 1234, "valid_entries": 1100, "size_mb": 5.2}

CacheClear[]  # Clear all

CacheClear[30]  # Clear entries older than 30 days
```

### 6. Canonical Property Schema

**Mapping layer for stable property names:**

```python
"BirthDate"    → P569
"Occupation"   → P106
"AtomicNumber" → P1086
```

**50+ properties across domains:**
- People (15)
- Geography (10)
- Physical properties (8)
- Creative works (10)
- Chemistry (7)
- Organizations (5)

Add custom mappings:
```python
from mikoshilang.knowledge_full import CANONICAL_PROPERTIES
CANONICAL_PROPERTIES["MyProperty"] = "P12345"
```

### 7. Domain Packs ⭐ NEW

**Extensible knowledge sources beyond Wikidata/Wikipedia.**

#### PubChem Pack (Chemistry)

```python
PackSearch["pubchem", "caffeine"]
# Returns: [{"id": "CID2519", "label": "Compound 2519", ...}]

PackValue["pubchem", "2519", "MolecularFormula"]
# Returns: {"value": "C8H10N4O2", "license": "Public Domain"}

PackValue["pubchem", "2519", "MolecularWeight"]
# Returns: {"value": {"amount": "194.19", "unit": "g/mol"}}
```

**Built-in packs:**
- ✅ **PubChem** (chemistry) — Public Domain
- ✅ **Crossref** (scholarly papers) — Metadata CC0
- ✅ **OpenAlex** (scholarly graph) — CC0
- ✅ **GeoNames** (geographic data) — CC BY 4.0
- ✅ **World Bank** (economic indicators) — CC BY 4.0

#### Creating Custom Packs

```python
from mikoshilang.knowledge_full import KnowledgePack

class MyDomainPack(KnowledgePack):
    def search(self, query: str, **kwargs):
        # Your search implementation
        return [{"id": "...", "label": "..."}]
    
    def get_value(self, entity_id: str, property: str, **kwargs):
        # Your property retrieval
        return {"value": "...", "source": "..."}

# Register pack
from mikoshilang.knowledge_full import _KNOWLEDGE_PACKS
_KNOWLEDGE_PACKS["mydomain"] = MyDomainPack()
```

### 8. License Compliance

**Full attribution + license tracking:**

```python
result = EntityValue["Q42", "BirthDate"]
# result["license"] == "CC0" (Wikidata)
# result["source"] == "https://www.wikidata.org/wiki/Q42"

result = WikipediaText["Douglas Adams"]
# result["license"] == "CC BY-SA 3.0 (Wikipedia)"
# result["attribution"] == "Source: Wikipedia contributors"
```

**License types:**
- **Wikidata:** CC0 (public domain)
- **Wikipedia:** CC BY-SA 3.0 (attribution required)
- **PubChem:** Public Domain
- **Custom packs:** Set your own

## Complete Examples

### Research Workflow

```python
# 1. Find entity
entities = EntitySearch["CRISPR"]
entity = entities[0]

# 2. Get all properties
value = EntityValue[entity["id"], "DiscoveryDate"]

# 3. Find related entities
inventors = EntityRelationships[entity["id"], "Inventor", 1, 10]

# 4. Get explanatory text
summary = WikipediaText[entity["label"]]

# 5. Get domain-specific data (if available)
# paper = PackSearch["crossref", entity["label"]]
```

### Time-Travel Fact-Checking

```python
# How was entity described in 2015?
snapshot_2015 = KnowledgeAsOf["Q42", "2015-01-01", "wikidata"]

# What were the occupations listed?
occupations_2015 = EntityValue["Q42", "Occupation", as_of="2015-01-01"]

# Compare with current
occupations_now = EntityValue["Q42", "Occupation"]
```

### Chemistry Research

```python
# Search PubChem
compounds = PackSearch["pubchem", "aspirin"]
aspirin = compounds[0]

# Get properties
formula = PackValue["pubchem", aspirin["id"], "MolecularFormula"]
weight = PackValue["pubchem", aspirin["id"], "MolecularWeight"]
smiles = PackValue["pubchem", aspirin["id"], "CanonicalSMILES"]

# Cross-reference with Wikidata
wikidata_results = EntitySearch["aspirin", "chemical compound"]
wikidata_aspirin = wikidata_results[0]

# Get Wikidata properties
wikidata_formula = EntityValue[wikidata_aspirin["id"], "ChemicalFormula"]
```

### Natural Language Interface

```python
# Let LLM plan the query
plan = InterpretQuery["Who wrote The Hitchhiker's Guide to the Galaxy?"]

# Execute plan steps:
# 1. EntitySearch["The Hitchhiker's Guide to the Galaxy", "book"]
# 2. EntityValue[book_id, "Author"]
# 3. Return author entity

# Or use with Anthropic Claude:
plan = InterpretQuery[
    "What were Douglas Adams' occupations in 2010?",
    llm_provider="anthropic",
    api_key="sk-..."
]
```

## Performance

### Caching Impact

| Operation | First Call | Cached Call | Speedup |
|-----------|-----------|-------------|---------|
| EntitySearch | ~200ms | <1ms | 200x |
| EntityValue | ~150ms | <1ms | 150x |
| WikipediaText | ~180ms | <1ms | 180x |
| EntityRelationships | ~400ms | <1ms | 400x |

### Cache Statistics

```python
stats = CacheStats[]
# {
#   "total_entries": 1523,
#   "valid_entries": 1498,
#   "expired_entries": 25,
#   "size_mb": 8.4,
#   "cache_file": "/home/user/.mikoshilang/knowledge_cache/knowledge.db"
# }
```

### Cache Cleanup

```python
# Manual cleanup
CacheClear[]  # Clear all
CacheClear[7]  # Clear entries older than 7 days

# Automatic cleanup on next query after expiry
# (expired entries are not returned)
```

## Migration from MVP

If you're using the MVP knowledge layer (`knowledge.py`), the full version is **backward-compatible**:

```python
# Old MVP functions still work
from mikoshilang.knowledge import EntitySearch, EntityValue, WikipediaSummary

# New full functions add features
from mikoshilang.knowledge_full import (
    EntitySearch,          # Same + caching
    EntityValue,           # Same + time-travel
    EntityRelationships,   # NEW
    WikipediaText,         # Replaces WikipediaSummary + sections
    KnowledgeAsOf,         # NEW
    InterpretQuery,        # NEW
    PackSearch,            # NEW
    PackValue,             # NEW
    CacheStats,            # NEW
    CacheClear,            # NEW
)
```

## API Reference

### Core Functions

| Function | Args | Returns | Cache TTL |
|----------|------|---------|-----------|
| `EntitySearch[query, type?, limit?]` | query, type (opt), limit (opt) | List[Entity] | 7d |
| `EntityValue[entity, property, asof?]` | entity QID/dict, property name, as-of date (opt) | PropertyValue | 7d |
| `EntityRelationships[entity, relation, depth?, limit?]` | entity QID, relation name, depth (1-3), limit | RelationshipGraph | 7d |
| `WikipediaText[title, section?, asof?, sentences?]` | title, section (opt), as-of (opt), sentences (opt) | Text | 24h |
| `KnowledgeAsOf[entity, date, source?]` | entity/title, ISO date, source | Snapshot | — |

### Domain Pack Functions

| Function | Args | Returns |
|----------|------|---------|
| `PackSearch[pack, query, ...]` | pack name, query, pack-specific args | List[Entity] |
| `PackValue[pack, entity, property, ...]` | pack name, entity ID, property | PropertyValue |

### LLM Interpreter

| Function | Args | Returns |
|----------|------|---------|
| `InterpretQuery[query, provider?, key?]` | natural language, LLM provider (opt), API key (opt) | QueryPlan + Results |

### Cache Management

| Function | Args | Returns |
|----------|------|---------|
| `CacheStats[]` | — | Statistics dict |
| `CacheClear[days?]` | older than N days (opt) | Cleared count |

## Troubleshooting

### Cache location

```bash
ls ~/.mikoshilang/knowledge_cache/
# knowledge.db
```

### Cache size growing

```python
# Check size
stats = CacheStats[]
print(f"Cache: {stats['size_mb']} MB")

# Clear old entries
CacheClear[30]  # Older than 30 days
```

### API rate limits

- Wikidata: No official limit, but use caching
- Wikipedia: No official limit
- PubChem: 5 requests/second, 400/minute

All APIs respect `User-Agent` header.

### Missing data

If a property returns `None`:
1. Check entity has that property on Wikidata
2. Try canonical property name vs PID directly
3. Check spelling of property name

## Extending

### Add a new canonical property

```python
from mikoshilang.knowledge_full import CANONICAL_PROPERTIES
CANONICAL_PROPERTIES["GivenName"] = "P735"
```

### Add a new domain pack

```python
from mikoshilang.knowledge_full import KnowledgePack, _KNOWLEDGE_PACKS

class CrossrefPack(KnowledgePack):
    API_BASE = "https://api.crossref.org"
    
    def search(self, query: str, limit: int = 5):
        # Implement Crossref search
        pass
    
    def get_value(self, doi: str, property: str):
        # Implement property retrieval
        pass

_KNOWLEDGE_PACKS["crossref"] = CrossrefPack()
```

### Custom LLM provider

```python
# Set API key in environment
export ANTHROPIC_API_KEY="sk-..."

# Use in queries
InterpretQuery["your question", "anthropic"]
```

## License

Knowledge layer code: Apache 2.0

Data sources:
- Wikidata: CC0 (public domain)
- Wikipedia: CC BY-SA 3.0 (attribution required)
- PubChem: Public Domain

#### Crossref Pack (Scholarly Papers)

```python
# Search papers
papers = PackSearch["crossref", "machine learning"]
# Returns: [{"id": "10.1145/...", "label": "Paper Title", ...}]

# Get metadata
title = PackValue["crossref", "10.1145/3422622", "Title"]
authors = PackValue["crossref", "10.1145/3422622", "Authors"]
citations = PackValue["crossref", "10.1145/3422622", "CitationCount"]
```

**Properties:**
- Title, Authors, PublicationDate, Publisher
- Abstract, CitationCount
- DOI (identifier)

**License:** Metadata is CC0 (public domain)

#### OpenAlex Pack (Scholarly Graph)

```python
# Search works
works = PackSearch["openalex", "deep learning"]
# Returns: [{"id": "W2741809807", "label": "...", ...}]

# Get metadata
title = PackValue["openalex", "W2741809807", "Title"]
citations = PackValue["openalex", "W2741809807", "CitationCount"]
concepts = PackValue["openalex", "W2741809807", "Concepts"]
oa_url = PackValue["openalex", "W2741809807", "OpenAccessURL"]
```

**Properties:**
- Title, Authors, PublicationDate
- CitationCount, Concepts (topics)
- OpenAccessURL

**License:** CC0 (public domain)

#### GeoNames Pack (Geographic Data)

```python
# Search places
places = PackSearch["geonames", "London"]
# Returns: [{"id": "2643743", "label": "London", ...}]

# Get metadata
country = PackValue["geonames", "2643743", "Country"]
population = PackValue["geonames", "2643743", "Population"]
lat = PackValue["geonames", "2643743", "Latitude"]
lon = PackValue["geonames", "2643743", "Longitude"]
```

**Properties:**
- Name, Country, Population
- Latitude, Longitude, Elevation
- Timezone

**License:** CC BY 4.0 (attribution required)

**Note:** Uses demo account (limited). Get free account at http://www.geonames.org/login

#### World Bank Pack (Economic Data)

```python
# Search indicators
indicators = PackSearch["worldbank", "GDP"]
# Returns: [{"id": "NY.GDP.MKTP.CD", "label": "GDP (current US$)", ...}]

# Get data
gdp = PackValue["worldbank", "NY.GDP.MKTP.CD", "Value", country="US"]
date = PackValue["worldbank", "NY.GDP.MKTP.CD", "Date", country="US"]
```

**Properties:**
- Value, Date, Country, Unit

**Parameters:**
- `country` — ISO 2-letter code or "all" for world

**License:** CC BY 4.0 (attribution required)

### All Packs Summary

| Pack | Domain | API | License | Rate Limit |
|------|--------|-----|---------|------------|
| PubChem | Chemistry | NIH/NLM | Public Domain | 5 req/sec |
| Crossref | Papers | Crossref.org | Metadata: CC0 | None (polite) |
| OpenAlex | Scholarly | OpenAlex.org | CC0 | None |
| GeoNames | Geography | GeoNames.org | CC BY 4.0 | Demo account |
| World Bank | Economics | WorldBank.org | CC BY 4.0 | None |
