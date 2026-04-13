#!/usr/bin/env python
"""Quick debug script to test if the inference loop works."""

import asyncio
import json
import os
from openai import OpenAI
from models import SqlSurgeonActionType

# Test JSON parsing
test_responses = [
    '{"action_type": "schema"}',
    '[START]{"action_type": "explain", "query": "SELECT * FROM users"}[END]',
    '```json\n{"action_type": "think", "thoughts": "Let me analyze this"}\n```',
    'I think we should use {"action_type": "submit", "query": "SELECT id FROM users", "confidence": 0.95}',
]

def _parse_candidates(text: str):
    """Copy of function from inference.py for testing."""
    text = (text or "").strip()
    if not text:
        return []
    seen = []
    
    def add(s: str) -> None:
        s = (s or "").strip()
        if s and s not in seen:
            seen.append(s)
    
    add(text)
    return seen

def _parse_action_segment(text: str):
    """Copy of function from inference.py for testing."""
    text = (text or "").strip()
    if not text:
        return None
    
    import re
    
    if "[START]" in text and "[END]" in text:
        chunk = text.split("[START]", 1)[1].split("[END]", 1)[0].strip()
        try:
            return json.loads(chunk)
        except json.JSONDecodeError as e:
            print(f"  [START/END] parse failed: {e}")
    
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if fence:
        try:
            return json.loads(fence.group(1).strip())
        except json.JSONDecodeError as e:
            print(f"  [Code fence] parse failed: {e}")
    
    start = text.find("{")
    if start >= 0:
        depth = 0
        for i, ch in enumerate(text[start:], start=start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError as e:
                        print(f"  [Brace matching] parse failed: {e}")
                        break
    
    return None

print("Testing JSON parsing from LLM responses:\n")

for i, response in enumerate(test_responses, 1):
    print(f"Test {i}: {response[:60]}...")
    for candidate in _parse_candidates(response):
        parsed = _parse_action_segment(candidate)
        if parsed:
            print(f"  ✓ Parsed: {parsed}")
            action_type = parsed.get("action_type", "unknown")
            print(f"  Action type: {action_type}")
        else:
            print(f"  ✗ Failed to parse")
    print()

# Test enum mapping
print("\nTesting action type enum mapping:\n")
test_types = ["schema", "explain", "think", "submit", "SCHEMA", "Explain", "invalid"]
for t in test_types:
    t_lower = t.lower().strip()
    mapped = None
    for enum_val in SqlSurgeonActionType:
        if enum_val.value == t_lower:
            mapped = enum_val
            break
    if mapped:
        print(f"✓ '{t}' → {mapped.name} ({mapped.value})")
    else:
        print(f"✗ '{t}' → defaults to THINK")
