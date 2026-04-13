#!/usr/bin/env python
"""Test LLM response parsing directly."""

import json
import os
from openai import OpenAI

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    print("ERROR: HF_TOKEN not set")
    exit(1)

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# Test prompt
system_prompt = """Respond with ONLY a JSON object. Nothing else.
{
  "action_type": "schema"
}
That's it. Just that JSON. No markdown, no explanation."""

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "Start the optimization."},
]

print("Calling LLM...")
completion = client.chat.completions.create(
    model=MODEL_NAME,
    messages=messages,
    temperature=0.1,
    max_tokens=500,
)

response_text = completion.choices[0].message.content
print(f"\n=== RAW LLM RESPONSE ===")
print(repr(response_text))
print(f"\n=== FORMATTED ===")
print(response_text)

# Try to parse it
print(f"\n=== PARSE ATTEMPT ===")
try:
    # Try direct JSON parse
    parsed = json.loads(response_text)
    print(f"✓ Direct parse OK: {parsed}")
except json.JSONDecodeError as e:
    print(f"✗ Direct parse failed: {e}")
    
    # Try to extract JSON
    import re
    match = re.search(r'\{[^{}]*\}', response_text)
    if match:
        try:
            parsed = json.loads(match.group())
            print(f"✓ Extracted JSON: {parsed}")
        except json.JSONDecodeError as e2:
            print(f"✗ Extracted parse failed: {e2}")
            print(f"  Attempted: {match.group()}")
    else:
        print("✗ No JSON found in response")
