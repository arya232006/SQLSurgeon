import requests


BASE_URL = "https://aryadeep232006-sqlsurgeon.hf.space"


def main() -> None:
    reset = requests.post(f"{BASE_URL}/reset", json={"task_id": "filter_scan"}, timeout=30)
    reset.raise_for_status()
    reset_payload = reset.json()
    print("Reset keys:", list(reset_payload.keys()))
    observation = reset_payload["observation"]
    print("Task:", observation["task_id"])
    print("Original query preview:", observation.get("original_query", "")[:120], "...")

    action = {
        "action": {
            "action_type": "think",
            "query": "",
            "thoughts": "Review schema, then explain plan.",
        }
    }
    step = requests.post(f"{BASE_URL}/step", json=action, timeout=30)
    step.raise_for_status()
    step_payload = step.json()
    print("Step reward:", step_payload["reward"])
    print("Done:", step_payload["done"])

    state = requests.get(f"{BASE_URL}/state", timeout=30)
    state.raise_for_status()
    state_payload = state.json()
    print("State keys:", list(state_payload.keys()))


if __name__ == "__main__":
    main()
