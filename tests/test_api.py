from fastapi.testclient import TestClient
import api


def valid_payload():
    return {
        "sector_group": "Consumer & Retail",
        "stage_bucket": "Seed",
        "city_group": "Bengaluru",
        "amount_bin": "$1M-5M",
        "amount_log10": 6.5,
        "company_age": 3,
        "top_k": 5,
    }


def test_home(client):
    response = client.get("/")
    assert response.status_code == 200


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] in ["healthy", "degraded"]


def test_options(client):
    response = client.get("/options")
    assert response.status_code == 200
    data = response.json()
    assert "Consumer & Retail" in data["sector_group"]
    assert "Seed" in data["stage_bucket"]
    assert "Bengaluru" in data["city_group"]
    assert "$1M-5M" in data["amount_bin"]


def test_invalid_company_age(client):
    payload = valid_payload()
    payload["company_age"] = -1
    response = client.post("/rank", json=payload)
    assert response.status_code == 422


def test_invalid_sector(client):
    payload = valid_payload()
    payload["sector_group"] = "Technology"
    response = client.post("/rank", json=payload)
    assert response.status_code == 422


def test_invalid_stage(client):
    payload = valid_payload()
    payload["stage_bucket"] = "Pre-seed"
    response = client.post("/rank", json=payload)
    assert response.status_code == 422


def test_invalid_city(client):
    payload = valid_payload()
    payload["city_group"] = "Invalid City"
    response = client.post("/rank", json=payload)
    assert response.status_code == 422


def test_invalid_amount_bin(client):
    payload = valid_payload()
    payload["amount_bin"] = "1M-5M"
    response = client.post("/rank", json=payload)
    assert response.status_code == 422


def test_amount_log10_null(client):
    payload = valid_payload()
    payload["amount_log10"] = None
    response = client.post("/rank", json=payload)
    assert response.status_code == 200
    assert len(response.json()["ranked_investors"]) == 5



def test_real_ranker_flow():
    """Exercise /rank with the shipped trained checkpoint, not a fake ranker."""
    ranker = api.load_ranker()
    api.app.state.ranker = ranker
    with TestClient(api.app) as real_client:
        response = real_client.post("/rank", json=valid_payload())
    assert response.status_code == 200
    rows = response.json()["ranked_investors"]
    assert len(rows) == 5
    assert all(0 <= row["score"] <= 100 for row in rows)
    assert all("model_probability" in row for row in rows)
