"""
Integration layer for full knowledge system into MikoshiLang evaluator.
"""

from .rules import RuleDelayed
from .pattern import Blank
from .expr import Expr
from .knowledge_full import (
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
)


def _extract_value(val):
    """Extract Python value from Expr or return as-is."""
    if isinstance(val, Expr):
        if val.head == "String" and len(val.args) == 1:
            return str(val.args[0])
        elif val.head == "Integer" and len(val.args) == 1:
            return int(val.args[0])
        elif len(val.args) == 0:
            return str(val.head)
        return str(val)
    return val


def register_full():
    """Register full knowledge layer functions."""
    
    rules = [
        # Entity Search
        RuleDelayed(
            Expr("EntitySearch", Blank("query")),
            lambda match: EntitySearch(_extract_value(match["query"]))
        ),
        RuleDelayed(
            Expr("EntitySearch", Blank("query"), Blank("type")),
            lambda match: EntitySearch(
                _extract_value(match["query"]),
                _extract_value(match["type"])
            )
        ),
        RuleDelayed(
            Expr("EntitySearch", Blank("query"), Blank("type"), Blank("limit")),
            lambda match: EntitySearch(
                _extract_value(match["query"]),
                _extract_value(match["type"]),
                _extract_value(match["limit"])
            )
        ),
        
        # Entity Value
        RuleDelayed(
            Expr("EntityValue", Blank("entity"), Blank("property")),
            lambda match: EntityValue(
                _extract_value(match["entity"]),
                _extract_value(match["property"])
            )
        ),
        RuleDelayed(
            Expr("EntityValue", Blank("entity"), Blank("property"), Blank("asof")),
            lambda match: EntityValue(
                _extract_value(match["entity"]),
                _extract_value(match["property"]),
                _extract_value(match["asof"])
            )
        ),
        
        # Entity Relationships (NEW)
        RuleDelayed(
            Expr("EntityRelationships", Blank("entity"), Blank("relation")),
            lambda match: EntityRelationships(
                _extract_value(match["entity"]),
                _extract_value(match["relation"])
            )
        ),
        RuleDelayed(
            Expr("EntityRelationships", Blank("entity"), Blank("relation"), Blank("depth")),
            lambda match: EntityRelationships(
                _extract_value(match["entity"]),
                _extract_value(match["relation"]),
                _extract_value(match["depth"])
            )
        ),
        RuleDelayed(
            Expr("EntityRelationships", Blank("entity"), Blank("relation"), Blank("depth"), Blank("limit")),
            lambda match: EntityRelationships(
                _extract_value(match["entity"]),
                _extract_value(match["relation"]),
                _extract_value(match["depth"]),
                _extract_value(match["limit"])
            )
        ),
        
        # Wikipedia Text
        RuleDelayed(
            Expr("WikipediaText", Blank("title")),
            lambda match: WikipediaText(_extract_value(match["title"]))
        ),
        RuleDelayed(
            Expr("WikipediaText", Blank("title"), Blank("sentences")),
            lambda match: WikipediaText(
                _extract_value(match["title"]),
                sentences=_extract_value(match["sentences"])
            )
        ),
        RuleDelayed(
            Expr("WikipediaText", Blank("title"), Blank("section"), Blank("sentences")),
            lambda match: WikipediaText(
                _extract_value(match["title"]),
                _extract_value(match["section"]),
                sentences=_extract_value(match["sentences"])
            )
        ),
        
        # Knowledge As Of (NEW)
        RuleDelayed(
            Expr("KnowledgeAsOf", Blank("entity"), Blank("date")),
            lambda match: KnowledgeAsOf(
                _extract_value(match["entity"]),
                _extract_value(match["date"])
            )
        ),
        RuleDelayed(
            Expr("KnowledgeAsOf", Blank("entity"), Blank("date"), Blank("source")),
            lambda match: KnowledgeAsOf(
                _extract_value(match["entity"]),
                _extract_value(match["date"]),
                _extract_value(match["source"])
            )
        ),
        
        # LLM Interpreter (NEW)
        RuleDelayed(
            Expr("InterpretQuery", Blank("query")),
            lambda match: InterpretQuery(_extract_value(match["query"]))
        ),
        RuleDelayed(
            Expr("InterpretQuery", Blank("query"), Blank("provider")),
            lambda match: InterpretQuery(
                _extract_value(match["query"]),
                _extract_value(match["provider"])
            )
        ),
        
        # Domain Packs (NEW)
        RuleDelayed(
            Expr("PackSearch", Blank("pack"), Blank("query")),
            lambda match: PackSearch(
                _extract_value(match["pack"]),
                _extract_value(match["query"])
            )
        ),
        RuleDelayed(
            Expr("PackValue", Blank("pack"), Blank("entity"), Blank("property")),
            lambda match: PackValue(
                _extract_value(match["pack"]),
                _extract_value(match["entity"]),
                _extract_value(match["property"])
            )
        ),
        
        # Cache Management
        RuleDelayed(
            Expr("CacheClear"),
            lambda match: CacheClear()
        ),
        RuleDelayed(
            Expr("CacheClear", Blank("days")),
            lambda match: CacheClear(_extract_value(match["days"]))
        ),
        RuleDelayed(
            Expr("CacheStats"),
            lambda match: CacheStats()
        ),
    ]
    
    return rules
