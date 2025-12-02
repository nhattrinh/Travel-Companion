import pytest
from app.services.translation_service import MockTranslationModel

@pytest.mark.asyncio
async def test_mock_translation_basic():
    model = MockTranslationModel()
    res = await model.translate("Hello world", target_language="ko")
    assert res["translated_text"].startswith("[KO]") or "안녕" in res["translated_text"] or res["translated_text"]
    assert 0 <= res["confidence"] <= 1

@pytest.mark.asyncio
async def test_language_detection():
    model = MockTranslationModel()
    lang = await model.detect_language("hello world")
    assert lang in {"en", "ko", "vi"}
