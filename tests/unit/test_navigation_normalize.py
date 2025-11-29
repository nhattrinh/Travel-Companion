from app.services.navigation_service import NavigationService

def test_normalize_poi_distance():
    svc = NavigationService()
    dist = svc._haversine(35.68, 139.76, 35.69, 139.77)
    assert dist > 0
    assert dist < 10000  # reasonable within city
