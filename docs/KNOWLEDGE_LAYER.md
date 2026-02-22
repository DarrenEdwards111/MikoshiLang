# MikoshiLang Knowledge Layer

**Wolfram-style entity framework** with structured facts from Wikidata and Wikipedia.

## Architecture

Unlike shallow "Wikipedia integration", MikoshiLang implements a proper knowledge layer:

1. **Entity layer** (Wikidata) — structured facts, QIDs, properties, relationships
2. **Text layer** (Wikipedia) — summaries, context, citations  
3. **LLM role** — orchestration only, not truth
4. **Deterministic tools** — no hallucination

## Core Functions

### EntitySearch

Search for entities in Wikidata.

```python
# Basic search
EntitySearch["Douglas Adams"]
# Returns: [{"id": "Q42", "label": "Douglas Adams", 
#            "description": "British science fiction writer...", ...}]

# With type hint
EntitySearch["Moon", "Film"]

# With result limit
EntitySearch["Einstein", "Person", 10]
```

**Returns:**
- `id`: Wikidata QID (e.g., "Q42")
- `label`: Primary name
- `description`: Short description
- `url`: Wikidata URL
- `retrieved`: Timestamp
- `source`: "Wikidata (CC0)"

### EntityValue

Get a specific property value for an entity.

```python
# Canonical property names
EntityValue["Q42", "BirthDate"]
# Returns: {"value": "1952-03-11", "property": "BirthDate", "pid": "P569", ...}

EntityValue["Q42", "Occupation"]
# Returns: {"value": [{linked entities}], ...}

# Or use Wikidata PIDs directly
EntityValue["Q42", "P569"]  # Birth date
```

**Canonical Properties:**
- `BirthDate`, `DeathDate`, `BirthPlace`
- `Occupation`, `Country`, `Population`
- `Mass`, `Height`, `Coordinate`
- `Director`, `Author`, `Publisher`, `PublicationDate`
- `ISBN`, `DOI`
- `ChemicalFormula`, `AtomicNumber`, `MolarMass`
- And 15+ more...

### EntityProperties

Get all available properties for an entity.

```python
EntityProperties["Q42"]
# Returns: {
#   "properties": {
#     "BirthDate": "1952-03-11",
#     "Occupation": [...],
#     ...
#   },
#   "source": "https://www.wikidata.org/wiki/Q42",
#   "retrieved": "2026-02-22T..."
# }
```

### WikipediaSummary

Get Wikipedia article summary (lead section).

```python
WikipediaSummary["Douglas Adams"]
# Returns: {
#   "summary": "Douglas Noel Adams was an English author...",
#   "title": "Douglas Adams",
#   "url": "https://en.wikipedia.org/wiki/Douglas_Adams",
#   "license": "CC BY-SA 3.0 (Wikipedia)",
#   "attribution": "Source: Wikipedia contributors"
# }

# Limit to N sentences
WikipediaSummary["Python (programming language)", 2]
```

### Disambiguate

Find disambiguation candidates for ambiguous queries.

```python
Disambiguate["Moon"]
# Returns: {
#   "candidates": [
#     {"id": "Q405", "description": "natural satellite of Earth"},
#     {"id": "Q219562", "description": "2009 film by Duncan Jones"},
#     ...
#   ],
#   "has_disambiguation_page": true,
#   "note": "Multiple entities found - please refine..."
# }

# With context hint
Disambiguate["Moon", "astronomy"]
```

### Entity (Convenience)

Quick entity lookup.

```python
Entity["Douglas Adams"]
# Returns: {"id": "Q42", "label": "Douglas Adams", ...}

Entity["Person", "Douglas Adams"]  # With type hint
```

## Provenance & Licensing

All knowledge results include:
- **Source URL** — where the data came from
- **Retrieved timestamp** — when it was fetched
- **License** — CC0 (Wikidata) or CC BY-SA 3.0 (Wikipedia)
- **Attribution** — required for Wikipedia text

## Caching

Entity lookups and facts are cached (LRU cache, 1000 entities max) for performance.

## Example Workflows

### Fact Lookup
```python
person = EntitySearch["Douglas Adams"][0]
birth_date = EntityValue[person["id"], "BirthDate"]
summary = WikipediaSummary[person["label"]]
```

### Research Assistant
```python
# Find entity
entities = EntitySearch["CRISPR", "Biology"]
entity = entities[0]

# Get all properties
props = EntityProperties[entity["id"]]

# Get explanatory text
summary = WikipediaSummary[entity["label"]]
```

### Citation-Ready Output
```python
result = EntityValue["Q42", "BirthDate"]
# result["source"] → https://www.wikidata.org/wiki/Q42
# result["retrieved"] → 2026-02-22T16:15:50Z
# result["license"] → CC0 (Wikidata)
```

## API Compliance

- **User-Agent** header set for all Wikimedia API requests
- **Rate limiting** respected (cached results reduce load)
- **License compliance** — CC0 (Wikidata), CC BY-SA 3.0 (Wikipedia text)

## Extending the Knowledge Layer

Add domain-specific packs:
- PubChem (chemistry)
- Crossref (papers)
- OpenAlex (scholarly graph)
- World Bank (economics)
- GeoNames (places)

Each pack implements the same 5 core tools for consistency.
