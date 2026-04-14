"""Quick test to see what the LLM actually returns."""
import os, json
from openai import OpenAI

client = OpenAI(
    base_url=os.getenv("API_BASE_URL", "https://router.huggingface.co/v1"),
    api_key=os.getenv("HF_TOKEN"),
)

model = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

# Test 1: Simple JSON output
print("=== TEST 1: Basic JSON output ===")
resp = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "You are an SQL optimizer. Respond with ONLY a valid JSON object."},
        {"role": "user", "content": 'Return this exact JSON: {"action_type":"schema"}'},
    ],
    temperature=0.1,
    max_tokens=256,
    response_format={"type": "json_object"},
)
msg = resp.choices[0].message
print(f"content type: {type(msg.content)}")
print(f"content repr: {repr(msg.content[:500])}")
print(f"finish_reason: {resp.choices[0].finish_reason}")
print()

# Test 2: With SQL optimization context
print("=== TEST 2: SQL optimization task ===")
resp2 = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "You are an SQL optimizer. Respond with ONLY a valid JSON object. Available actions: schema, explain, think, submit."},
        {"role": "user", "content": """Optimize this slow SQL query:
SELECT * FROM orders WHERE status = 'delivered'

Respond with a JSON action like:
{"action_type":"submit","query":"SELECT * FROM orders WHERE status = 'delivered'","confidence":0.9}
or {"action_type":"explain","query":"SELECT * FROM orders WHERE status = 'delivered'"}
"""},
    ],
    temperature=0.1,
    max_tokens=512,
    response_format={"type": "json_object"},
)
msg2 = resp2.choices[0].message
print(f"content type: {type(msg2.content)}")
print(f"content repr: {repr(msg2.content[:1000])}")
print(f"finish_reason: {resp2.choices[0].finish_reason}")

# Try to parse it
try:
    parsed = json.loads(msg2.content)
    print(f"parsed action_type: {parsed.get('action_type')}")
    print(f"parsed keys: {list(parsed.keys())}")
except Exception as e:
    print(f"Parse failed: {e}")
