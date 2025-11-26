"""
Basic test to verify the project setup is working correctly.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.core.models import ModelManager, ModelType, ModelStatus


def test_app_creation():
    """Test that the FastAPI app can be created successfully."""
    assert app is not None
    assert app.title == "Menu Translation Backend"


def test_root_endpoint():
    """Test the root endpoint returns expected response."""
    client = TestClient(app)
    response = client.get("/")
    
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data
    assert "status" in data
    assert data["status"] == "running"


def test_model_manager_creation():
    """Test that ModelManager can be created and initialized."""
    manager = ModelManager()
    assert manager is not None
    assert len(manager._models) == 0
    assert len(manager._model_locks) == len(ModelType)


@pytest.mark.asyncio
async def test_model_manager_status():
    """Test model manager status functionality."""
    manager = ModelManager()
    
    # Test getting status for non-existent model
    status = await manager.get_model_status(ModelType.OCR)
    assert status == ModelStatus.FAILED
    
    # Test getting all model statuses
    all_statuses = await manager.get_all_model_statuses()
    assert isinstance(all_statuses, dict)
    assert len(all_statuses) == len(ModelType)
    assert all(status is False for status in all_statuses.values())


if __name__ == "__main__":
    pytest.main([__file__])