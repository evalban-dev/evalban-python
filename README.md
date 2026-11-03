# evalban

Python client for [Evalban](https://evalban.com): project management for AI product teams, with your evals built in.
Send an eval run, and if the score lands below baseline Evalban files a bug with the failing examples, blocks the
merge and tells Slack.

```bash
pip install evalban
```

No dependencies beyond the Python standard library.

## Send an eval run

```python
import os
from evalban import Evalban

client = Evalban(api_key=os.environ["EVALBAN_API_KEY"], url=os.environ["EVALBAN_URL"])

res = client.log_run("listing-classifier", 0.871,
    version="ft-1027",
    failures=[{"input": "Grey sofa", "expected": "Furniture", "output": "Outdoor"}],
    slices={"furniture": 0.74},
    cost_per_call_usd=0.0004, latency_p95_ms=168)

if res["gate"] == "regressed":
    print("Filed", res["bug_key"])
```

`EVALBAN_URL` and your API key are on the **API keys** page in Evalban.

## Use your eval set

Cases promoted from fixed bugs come back in the eval set, so a fixed failure can't quietly return.

```python
cases = client.get_eval_cases("listing-classifier")
```

## Production feedback

```python
client.send_feedback("reply-drafts", [{"kind": "negative", "input": "...", "output": "..."}])
```

## Command line

```bash
evalban log --experiment listing-classifier --file results.json --format promptfoo --version ft-1027
```

Formats: `evalban`, `promptfoo`, `jsonl`, `csv`. Reads `EVALBAN_API_KEY` and `EVALBAN_URL` from the environment.
Exits with status 1 when the run regressed, so it can fail a CI job.

## License

MIT
