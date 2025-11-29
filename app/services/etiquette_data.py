"""Etiquette rules dataset (Phase 4 US2)."""

ETIQUETTE_RULES = {
    "transit": ["Queue in orderly lines", "Lower voice on trains", "No phone calls"],
    "restaurant": ["Shoes off at entrance (if indicated)", "Don't tip", "Use chopsticks properly"],
    "temple": ["Bow at entrance", "Quiet voice", "Photography may be restricted"],
}

def get_context_notes(category: str) -> list[str]:
    """
    Get etiquette rules for a specific category.
    
    Args:
        category: Category name (transit, restaurant, temple)
        
    Returns:
        List of etiquette rules for the category, empty list if not found
    """
    return ETIQUETTE_RULES.get(category, [])
