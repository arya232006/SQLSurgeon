"""Minimal test: what does the LLM actually return?"""
import os, json
from openai import OpenAI

client = OpenAI(
    base_url=os.getenv("API_BASE_URL", "https://router.huggingface.co/v1"),
    api_key=os.getenv("HF_TOKEN"),
)
model = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

system = """You are an SQL optimizer for SQLite. Respond with exactly ONE JSON object. No other text.

Available actions:
  {"action_type":"schema"}
  {"action_type":"explain","query":"SELECT ..."}
  {"action_type":"submit","query":"SELECT ...","confidence":0.95}
"""

user = """Optimize this slow SQL query:

SELECT sub.order_id, sub.customer_name, sub.total_amount, sub.city
FROM (
    SELECT o.id AS order_id, c.first_name || ' ' || c.last_name AS customer_name,
           o.total_amount, c.city, o.status, c.country
    FROM orders o, customers c WHERE o.customer_id = c.id
) sub
WHERE sub.status = 'delivered' AND sub.country = 'India' AND sub.total_amount > 500
ORDER BY sub.total_amount DESC;

Respond with a JSON action to optimize this query. For example:
{"action_type":"submit","query":"SELECT o.id AS order_id, c.first_name || ' ' || c.last_name AS customer_name, o.total_amount, c.city FROM orders o JOIN customers c ON o.customer_id = c.id WHERE o.status = 'delivered' AND c.country = 'India' AND o.total_amount > 500 ORDER BY o.total_amount DESC","confidence":0.95}
"""

print(f"Model: {model}")
print(f"API: {client.base_url}")
print()

# Test WITHOUT json mode
print("=== WITHOUT response_format ===")
try:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.1,
        max_tokens=1024,
    )
    content = resp.choices[0].message.content
    print(f"finish_reason: {resp.choices[0].finish_reason}")
    print(f"content:\n{content}")
    print()
    try:
        parsed = json.loads(content)
        print(f"Parsed OK: {list(parsed.keys())}")
    except:
        print("Not valid JSON")
except Exception as e:
    print(f"Error: {e}")

print()
print("=== WITH response_format=json_object ===")
try:
    resp2 = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.1,
        max_tokens=1024,
        response_format={"type": "json_object"},
    )
    content2 = resp2.choices[0].message.content
    print(f"finish_reason: {resp2.choices[0].finish_reason}")
    print(f"content:\n{content2}")
    print()
    try:
        parsed2 = json.loads(content2)
        print(f"Parsed OK: {list(parsed2.keys())}")
        print(f"action_type: {parsed2.get('action_type')}")
    except:
        print("Not valid JSON")
except Exception as e:
    print(f"Error: {e}")
