from app.services.etiquette_data import get_context_notes

def test_context_builder():
    notes = get_context_notes("transit")
    assert isinstance(notes, list)
    assert len(notes) > 0
    unknown = get_context_notes("unknown_cat")
    assert unknown == []
