#!/usr/bin/env python
"""Verify parsing and enum mapping logic works correctly."""

import json
from models import SqlSurgeonActionType

def test_action_type_mapping():
    """Test that string action types map correctly to enums."""
    test_cases = [
        ("schema", SqlSurgeonActionType.CHECK_SCHEMA),
        ("explain", SqlSurgeonActionType.RUN_EXPLAIN),
        ("think", SqlSurgeonActionType.THINK),
        ("submit", SqlSurgeonActionType.SUBMIT),
        ("SCHEMA", SqlSurgeonActionType.CHECK_SCHEMA),  # Case insensitive
        ("Explain", SqlSurgeonActionType.RUN_EXPLAIN),
        ("  think  ", SqlSurgeonActionType.THINK),  # Whitespace trimmed
        ("invalid", SqlSurgeonActionType.THINK),  # Default to THINK
    ]
    
    for action_str, expected_enum in test_cases:
        # This is the logic from inference.py
        action_type_str = action_str.lower().strip()
        action_enum = SqlSurgeonActionType.THINK
        for enum_val in SqlSurgeonActionType:
            if enum_val.value == action_type_str:
                action_enum = enum_val
                break
        
        assert action_enum == expected_enum, f"Failed for '{action_str}': got {action_enum}, expected {expected_enum}"
        print(f"✓ '{action_str}' → {action_enum.name}")

def test_server_side_comparison():
    """Test that server-side string comparisons work."""
    test_cases = [
        ("schema", SqlSurgeonActionType.CHECK_SCHEMA, True),
        ("explain", SqlSurgeonActionType.RUN_EXPLAIN, True),
        ("think", SqlSurgeonActionType.THINK, True),
        ("submit", SqlSurgeonActionType.SUBMIT, True),
        ("invalid", SqlSurgeonActionType.CHECK_SCHEMA, False),
    ]
    
    for action_str, enum_val, should_match in test_cases:
        # This is the server-side logic
        a_type_str = str(action_str).strip().lower()
        enum_name = enum_val.value.lower()
        matches = (a_type_str == enum_name) or (action_str == enum_val)
        
        assert matches == should_match, f"Failed for '{action_str}' vs {enum_val}"
        status = "✓" if matches else "✗"
        print(f"{status} '{action_str}' vs {enum_val.name}: {matches}")

def test_json_serialization():
    """Test that observation data serializes properly."""
    obs_data = {
        "task_id": "filter_scan",
        "description": "Find orders with value > 1000",
        "original_query": "SELECT * FROM orders WHERE total > 1000",
        "actions_remaining": 15,
    }
    
    # This is what the client does
    json_str = json.dumps(obs_data, indent=2)
    
    # Verify it's valid JSON
    parsed = json.loads(json_str)
    assert parsed == obs_data
    print(f"✓ Observation serialized correctly ({len(json_str)} chars)")
    print(f"  Preview: {json_str[:100]}...")

def test_action_response_parsing():
    """Test that various JSON action formats parse correctly."""
    test_responses = [
        '{"action_type": "schema"}',
        '{"action_type": "explain", "query": "SELECT * FROM users"}',
        '[START]{"action_type": "think", "thoughts": "Let me analyze"}[END]',
        '```json\n{"action_type": "submit", "query": "SELECT id FROM users", "confidence": 0.95}\n```',
    ]
    
    from inference import _parse_action_from_llm_text, _normalize_parsed_action
    
    for response in test_responses:
        parsed = _parse_action_from_llm_text(response)
        assert parsed is not None, f"Failed to parse: {response[:50]}"
        normalized = _normalize_parsed_action(parsed)
        assert normalized is not None
        action_type = normalized.get("action_type")
        print(f"✓ Parsed response with action_type='{action_type}'")

if __name__ == "__main__":
    print("Testing action type mapping...")
    test_action_type_mapping()
    print()
    
    print("Testing server-side comparison...")
    test_server_side_comparison()
    print()
    
    print("Testing JSON serialization...")
    test_json_serialization()
    print()
    
    print("Testing action response parsing...")
    test_action_response_parsing()
    print()
    
    print("All tests passed! ✓")
