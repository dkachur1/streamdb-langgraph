from __future__ import annotations

from streamdb_langgraph.state_projection import project_client_state


def test_projects_active_workflow_from_graph_state() -> None:
    projected = project_client_state({"active_workflow": "weekly-audit", "tasks": {}})
    assert projected["active_workflow"] == "weekly-audit"


def test_omits_invalid_active_workflow_values() -> None:
    assert project_client_state({"active_workflow": 1})["active_workflow"] is None
    assert project_client_state({"active_workflow": ""})["active_workflow"] is None
    assert project_client_state({})["active_workflow"] is None


def test_projects_active_project_independently_of_workflow() -> None:
    projected = project_client_state(
        {"active_workflow": "weekly-audit", "active_project": "sedan-line", "tasks": {}}
    )
    assert projected["active_project"] == "sedan-line"
    assert projected["active_workflow"] == "weekly-audit"


def test_omits_invalid_active_project_values() -> None:
    assert project_client_state({"active_project": 1})["active_project"] is None
    assert project_client_state({"active_project": ""})["active_project"] is None
    assert project_client_state({})["active_project"] is None


def test_mode_field_is_no_longer_projected() -> None:
    projected = project_client_state({"mode": "simulation", "tasks": {}})
    assert "mode" not in projected


def test_projects_subagent_threads_keyed_by_tool_call_id() -> None:
    threads = {"tc-1": [{"id": "m1", "role": "assistant", "content": "hi"}]}
    projected = project_client_state({"subagent_threads": threads, "tasks": {}})
    assert projected["subagent_threads"] == threads


def test_subagent_threads_defaults_to_empty_when_absent_or_invalid() -> None:
    assert project_client_state({})["subagent_threads"] == {}
    assert project_client_state({"subagent_threads": None})["subagent_threads"] == {}
    assert project_client_state({"subagent_threads": []})["subagent_threads"] == {}
