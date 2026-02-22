"""
Knowledge layer rules for MikoshiLang evaluator integration.
"""

from .rules import RuleDelayed
from .pattern import Blank
from .expr import Expr
from .knowledge import (
    EntitySearch,
    EntityValue,
    EntityProperties,
    WikipediaSummary,
    Disambiguate,
    Entity,
)


def _extract_value(val):
    """Extract Python value from Expr or return as-is."""
    if isinstance(val, Expr):
        if val.head == "String" and len(val.args) == 1:
            return str(val.args[0])
        elif val.head == "Integer" and len(val.args) == 1:
            return int(val.args[0])
        # Try to get a simple value
        if len(val.args) == 0:
            return str(val.head)
        return str(val)
    return val


def register():
    """Register knowledge layer functions as evaluator rules."""
    
    rules = [
        # EntitySearch[query]
        RuleDelayed(
            Expr("EntitySearch", Blank("query")),
            lambda match: EntitySearch(_extract_value(match["query"]))
        ),
        
        # EntitySearch[query, type]
        RuleDelayed(
            Expr("EntitySearch", Blank("query"), Blank("type")),
            lambda match: EntitySearch(
                _extract_value(match["query"]),
                _extract_value(match["type"])
            )
        ),
        
        # EntitySearch[query, type, limit]
        RuleDelayed(
            Expr("EntitySearch", Blank("query"), Blank("type"), Blank("limit")),
            lambda match: EntitySearch(
                _extract_value(match["query"]),
                _extract_value(match["type"]),
                _extract_value(match["limit"])
            )
        ),
        
        # EntityValue[entity, property]
        RuleDelayed(
            Expr("EntityValue", Blank("entity"), Blank("property")),
            lambda match: EntityValue(
                _extract_value(match["entity"]),
                _extract_value(match["property"])
            )
        ),
        
        # EntityProperties[entity]
        RuleDelayed(
            Expr("EntityProperties", Blank("entity")),
            lambda match: EntityProperties(_extract_value(match["entity"]))
        ),
        
        # WikipediaSummary[title]
        RuleDelayed(
            Expr("WikipediaSummary", Blank("title")),
            lambda match: WikipediaSummary(_extract_value(match["title"]))
        ),
        
        # WikipediaSummary[title, sentences]
        RuleDelayed(
            Expr("WikipediaSummary", Blank("title"), Blank("sentences")),
            lambda match: WikipediaSummary(
                _extract_value(match["title"]),
                _extract_value(match["sentences"])
            )
        ),
        
        # Disambiguate[query]
        RuleDelayed(
            Expr("Disambiguate", Blank("query")),
            lambda match: Disambiguate(_extract_value(match["query"]))
        ),
        
        # Disambiguate[query, context]
        RuleDelayed(
            Expr("Disambiguate", Blank("query"), Blank("context")),
            lambda match: Disambiguate(
                _extract_value(match["query"]),
                _extract_value(match["context"])
            )
        ),
        
        # Entity[name]
        RuleDelayed(
            Expr("Entity", Blank("name")),
            lambda match: Entity(_extract_value(match["name"]))
        ),
        
        # Entity[type, name]
        RuleDelayed(
            Expr("Entity", Blank("type"), Blank("name")),
            lambda match: Entity(
                _extract_value(match["type"]),
                _extract_value(match["name"])
            )
        ),
    ]
    
    return rules
