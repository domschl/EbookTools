from typing import TypedDict

class CalibrePrefixes(TypedDict):
    de: dict[str, list[str]]
    en: dict[str, list[str]]
    
calibre_prefixes:CalibrePrefixes = {
    "de": {"prefixes": ["Der", "Die", "Das", "Ein", "Eine"]},
    "en": {"prefixes": ["The", "A", "An"]},
}
