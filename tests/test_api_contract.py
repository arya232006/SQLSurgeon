from fastapi.testclient import TestClient

from server.app import app


def test_openapi_docs_endpoints_are_available():
    client = TestClient(app)

    openapi = client.get("/openapi.json")
    assert openapi.status_code == 200
    assert "paths" in openapi.json()

    docs = client.get("/docs")
    assert docs.status_code == 200
    # Keep docs proxy-safe (no absolute /openapi.json fetch).
    assert "openapi.json" in docs.text
    assert "/openapi.json" not in docs.text


def test_health_and_schema_contracts():
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json().get("status") == "healthy"

    schema = client.get("/schema")
    assert schema.status_code == 200
    payload = schema.json()
    assert "action" in payload
    assert "observation" in payload
    assert "state" in payload


def test_reset_step_and_state_contracts():
    client = TestClient(app)

    reset = client.post("/reset", json={"task_id": "filter_scan"})
    assert reset.status_code == 200
    reset_payload = reset.json()
    assert set(["observation", "reward", "done"]).issubset(reset_payload.keys())
    assert "task_id" in reset_payload["observation"]

    step = client.post(
        "/step",
        json={
            "action": {
                "action_type": "think",
                "query": "",
                "thoughts": "contract test call",
            }
        },
    )
    assert step.status_code == 200
    step_payload = step.json()
    assert set(["observation", "reward", "done"]).issubset(step_payload.keys())
    assert "actions_remaining" in step_payload["observation"]

    state = client.get("/state")
    assert state.status_code == 200
    state_payload = state.json()
    assert "episode_id" in state_payload
    assert "step_count" in state_payload
