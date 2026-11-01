
from fastapi.testclient import TestClient
from api import app

client = TestClient(app)


def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "SponsorMatch API is running!"


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_rank_success():
    payload = {
        "sector_group": "Technology",
        "stage_bucket": "Seed",
        "city_group": "Bengaluru",
        "amount_bin": "1M-5M",
        "amount_log10": 6.5,
        "company_age": 3,
        "top_k": 5
    }

    response = client.post("/rank", json=payload)

    assert response.status_code == 200

    data = response.json()
    assert "ranked_investors" in data
    assert len(data["ranked_investors"]) <= 5


def test_invalid_company_age():
    payload = {
        "sector_group": "Technology",
        "stage_bucket": "Seed",
        "city_group": "Bengaluru",
        "amount_bin": "1M-5M",
        "amount_log10": 6.5,
        "company_age": -1,
        "top_k": 5
    }

    response = client.post("/rank", json=payload)
    assert response.status_code == 422
