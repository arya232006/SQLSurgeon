from models import SqlSurgeonObservation, SqlSurgeonState


def test_observation_list_defaults_are_not_shared():
    obs_a = SqlSurgeonObservation()
    obs_b = SqlSurgeonObservation()

    obs_a.deceptive_hints.append("misleading hint")
    assert obs_b.deceptive_hints == []


def test_state_list_defaults_are_not_shared():
    state_a = SqlSurgeonState()
    state_b = SqlSurgeonState()

    state_a.attempt_history.append("attempt_1")
    assert state_b.attempt_history == []
