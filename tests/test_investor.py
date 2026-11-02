import api
import pitch_requests
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def investor_client(tmp_path, monkeypatch):
    monkeypatch.setattr(pitch_requests, "DB_PATH", tmp_path / "investor-test.sqlite3")
    monkeypatch.setattr(api, "load_ranker", lambda: None)
    pitch_requests.init_db()
    with TestClient(api.app) as test_client:
        yield test_client


def test_investor_preferences_matching_and_watchlist(investor_client):
    registered = investor_client.post("/pitch/register", json={
        "email": "investor@example.com",
        "password": "secure-sample-password",
        "role": "investor",
        "name": "Sample Ventures",
        "preferred_sectors": ["Energy & Climate", "Fintech"],
        "preferred_stages": ["Seed", "Series A"],
        "preferred_cities": ["Bengaluru"],
        "min_ticket_usd": 100_000,
        "max_ticket_usd": 1_000_000,
        "summary": "Early-stage climate and fintech investor.",
    })
    assert registered.status_code == 201

    login = investor_client.post("/pitch/login", json={
        "email": "investor@example.com",
        "password": "secure-sample-password",
    })
    assert login.status_code == 200
    headers = {"Authorization": f"Bearer {login.json()['access_token']}"}

    preferences = investor_client.get("/pitch/investor/preferences", headers=headers)
    assert preferences.status_code == 200
    assert preferences.json()["preferred_sectors"] == ["Energy & Climate", "Fintech"]

    updated = {**preferences.json(), "preferred_sectors": ["Fintech"]}
    assert investor_client.put("/pitch/investor/preferences", headers=headers, json=updated).status_code == 200
    saved_preferences = investor_client.get("/pitch/investor/preferences", headers=headers).json()
    assert saved_preferences["preferred_sectors"] == ["Fintech"]

    matches = investor_client.post("/match/startups", json={
        "preferred_sectors": saved_preferences["preferred_sectors"],
        "preferred_stages": saved_preferences["preferred_stages"],
        "preferred_cities": saved_preferences["preferred_cities"],
        "min_ticket_usd": saved_preferences["min_ticket_usd"],
        "max_ticket_usd": saved_preferences["max_ticket_usd"],
        "top_k": 5,
    })
    assert matches.status_code == 200
    startup = matches.json()["matched_startups"][0]

    saved = investor_client.post("/pitch/investor/watchlist", headers=headers, json={
        "company_name": startup["company_name"],
        "company_domain": startup["company_domain"],
        "details": startup,
    })
    assert saved.status_code == 201
    watchlist = investor_client.get("/pitch/investor/watchlist", headers=headers)
    assert watchlist.json()[0]["company_name"] == startup["company_name"]
    assert investor_client.delete(
        f"/pitch/investor/watchlist/{saved.json()['id']}", headers=headers
    ).status_code == 200
