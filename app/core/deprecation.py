"""Field deprecation mapping utility.

Task: T151
Supports dual field names during deprecation window.
"""
from typing import Any, Dict, List


class DeprecationMapper:
    def __init__(self, mappings: Dict[str, str]):
        """Create mapper with old->new field name mapping."""
        self._mappings = mappings

    def transform_outbound(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Add deprecated field names pointing to new values.

        Avoid duplicating large value objects.
        """
        result = dict(payload)
        for old, new in self._mappings.items():
            if new in payload and old not in result:
                result[old] = payload[new]
        return result

    def transform_inbound(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize inbound payload.

        Prefer new names; map old names to new when absent.
        """
        result = dict(payload)
        for old, new in self._mappings.items():
            if old in payload and new not in result:
                result[new] = payload[old]
        return result

    def audit(self, payload: Dict[str, Any]) -> List[str]:
        """Return list of deprecated fields present so caller can log usage."""
        return [old for old in self._mappings.keys() if old in payload]
