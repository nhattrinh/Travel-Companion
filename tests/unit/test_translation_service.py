import pytest
from app.services.translation_service import MockTranslationModel

@pytest.mark.asyncio
async def test_mock_translation_basic():
    model = MockTranslationModel()
    res = await model.translate("Hello world", target_language="ja")
    assert res["translated_text"].startswith("[JA]")
    assert 0 <= res["confidence"] <= 1

@pytest.mark.asyncio
async def test_language_detection():
    model = MockTranslationModel()
    lang = await model.detect_language("el rapido coche")
    assert lang in {"en","es","fr","de","it","pt","zh","ja","ko"}
