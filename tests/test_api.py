"""tests/test_api.py — FastAPI endpoint integration tests."""
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "JARVIS online"


def test_query_empty_text():
    resp = client.post("/query", json={"text": ""})
    assert resp.status_code == 400


def test_reset():
    resp = client.post("/query/reset")
    assert resp.status_code == 200
