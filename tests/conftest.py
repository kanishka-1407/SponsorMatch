
import pytest
from fastapi.testclient import TestClient
import api


@pytest.fixture
def client(monkeypatch):
    """Test client without starting the ML model or application lifespan."""

    def no_model_for_api_tests():
        raise RuntimeError("Model loading disabled in API tests")

    monkeypatch.setattr(api, "load_ranker", no_model_for_api_tests)

    test_client = TestClient(api.app)
    yield test_client
    test_client.close()
