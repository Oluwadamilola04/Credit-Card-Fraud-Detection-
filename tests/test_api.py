from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def _sample_transaction() -> dict[str, float]:
    return {
        "Time": 0.0,
        "V1": -1.0,
        "V2": -0.5,
        "V3": 0.2,
        "V4": 1.1,
        "V5": -0.2,
        "V6": 0.0,
        "V7": 0.8,
        "V8": -0.4,
        "V9": 0.3,
        "V10": -0.1,
        "V11": 0.5,
        "V12": -0.6,
        "V13": 0.7,
        "V14": -0.2,
        "V15": 0.3,
        "V16": -0.9,
        "V17": 0.1,
        "V18": -0.3,
        "V19": 0.2,
        "V20": 0.0,
        "V21": -0.1,
        "V22": 0.4,
        "V23": 0.2,
        "V24": -0.7,
        "V25": 0.5,
        "V26": 0.8,
        "V27": -0.2,
        "V28": 0.1,
        "Amount": 50.0,
    }


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_metadata_endpoint() -> None:
    response = client.get("/metadata")
    assert response.status_code == 200
    payload = response.json()
    assert "model_path" in payload
    assert "threshold" in payload
    assert "expected_features" in payload


def test_predict_single_transaction() -> None:
    response = client.post("/predict", json={"transaction": _sample_transaction()})
    assert response.status_code == 200
    payload = response.json()
    assert "predictions" in payload
    assert len(payload["predictions"]) == 1
    assert 0.0 <= float(payload["predictions"][0]["fraud_probability"]) <= 1.0
    assert isinstance(payload["predictions"][0]["is_fraud"], bool)
