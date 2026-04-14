const BASE_URL = "https://aryadeep232006-sqlsurgeon.hf.space";

async function main() {
  const resetResp = await fetch(`${BASE_URL}/reset`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ task_id: "filter_scan" }),
  });
  if (!resetResp.ok) {
    throw new Error(`Reset failed: ${resetResp.status}`);
  }
  const reset = await resetResp.json();
  console.log("reset keys:", Object.keys(reset));
  console.log("task:", reset.observation?.task_id);

  const stepResp = await fetch(`${BASE_URL}/step`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      action: {
        action_type: "think",
        query: "",
        thoughts: "Review schema and execution plan first.",
      },
    }),
  });
  if (!stepResp.ok) {
    throw new Error(`Step failed: ${stepResp.status}`);
  }
  const step = await stepResp.json();
  console.log("step reward:", step.reward, "done:", step.done);

  const stateResp = await fetch(`${BASE_URL}/state`);
  if (!stateResp.ok) {
    throw new Error(`State failed: ${stateResp.status}`);
  }
  const state = await stateResp.json();
  console.log("state keys:", Object.keys(state));
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
