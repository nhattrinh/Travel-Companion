"""Unit tests for DeprecationMapper.

Task: T152
"""
from app.core.deprecation import DeprecationMapper


def test_transform_outbound_adds_old_fields():
    mapper = DeprecationMapper({"oldName": "new_name"})
    payload = {"new_name": "value"}
    out = mapper.transform_outbound(payload)
    assert out["oldName"] == "value"
    assert out["new_name"] == "value"


def test_transform_inbound_normalizes_old():
    mapper = DeprecationMapper({"oldName": "new_name"})
    payload = {"oldName": "value"}
    norm = mapper.transform_inbound(payload)
    assert norm["new_name"] == "value"
    assert "oldName" in norm  # original preserved


def test_audit_detects_deprecated():
    mapper = DeprecationMapper({"legacy": "current"})
    payload = {"legacy": 1, "other": 2}
    deprecated = mapper.audit(payload)
    assert deprecated == ["legacy"]
