#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${1:-https://aryadeep232006-sqlsurgeon.hf.space}"

echo "Using BASE_URL=${BASE_URL}"
echo

echo "1) Reset environment"
RESET_RESPONSE="$(curl -sS -X POST "${BASE_URL}/reset" \
  -H "Content-Type: application/json" \
  -d '{"task_id":"filter_scan"}')"
echo "${RESET_RESPONSE}" | python -m json.tool
echo

echo "2) Take a THINK step"
STEP_RESPONSE="$(curl -sS -X POST "${BASE_URL}/step" \
  -H "Content-Type: application/json" \
  -d '{"action":{"action_type":"think","query":"","thoughts":"Inspect schema and query plan first"}}')"
echo "${STEP_RESPONSE}" | python -m json.tool
echo

echo "3) Inspect state"
curl -sS "${BASE_URL}/state" | python -m json.tool
