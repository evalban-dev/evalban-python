"""Evalban eval SDK for Python. Standard library only.

    from evalban import Evalban
    client = Evalban(api_key=os.environ["EVALBAN_API_KEY"], url=os.environ["EVALBAN_URL"])

    # Pull the eval set (includes cases promoted from fixed bugs), run it, send the results.
    cases = client.get_eval_cases("listing-classifier")
    res = client.log_run("listing-classifier", 0.871, version="ft-1027",
                         failures=[{"input": "Grey sofa", "expected": "Furniture", "output": "Outdoor"}],
                         slices={"furniture": 0.74, "electronics": 0.91},
                         scores={"hallucination_rate": 0.03},
                         examples=[{"key": c["key"], "passed": True} for c in cases])
    if res["gate"] == "regressed":
        print("Evalban:", res["bug_key"])

    # Production feedback
    client.send_feedback("reply-drafts", [{"kind": "negative", "input": "...", "output": "..."}])

Command line (import results from another tool and send them):
    python -m evalban log --experiment listing-classifier --file results.json --format promptfoo --version ft-1027
    Formats: evalban, promptfoo, jsonl, csv. Reads EVALBAN_API_KEY and EVALBAN_URL from the environment.
"""
import csv
import io
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

__all__ = ["Evalban", "EvalbanError", "load_results"]


class EvalbanError(Exception):
    def __init__(self, message, status):
        super().__init__(message)
        self.status = status


class Evalban:
    def __init__(self, api_key, url, timeout=15):
        if not api_key:
            raise ValueError("Evalban: api_key is required")
        if not url:
            raise ValueError("Evalban: url is required (your Supabase project URL)")
        self.api_key = api_key
        url = url.rstrip("/")
        self.endpoint = url if url.endswith("/eval-ingest") else url + "/functions/v1/eval-ingest"
        self.base = self.endpoint[: -len("/eval-ingest")]
        self.timeout = timeout

    def _request(self, url, payload=None):
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="GET" if payload is None else "POST", headers={
            "Authorization": "Bearer " + self.api_key,
            "Content-Type": "application/json",
        })
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            try:
                message = json.loads(e.read().decode("utf-8")).get("error")
            except Exception:
                message = None
            raise EvalbanError(message or "Evalban returned %d" % e.code, e.code) from None

    def log_run(self, experiment, score, version="", failures=None,
                cost_per_call_usd=None, latency_p95_ms=None, tokens_per_call=None,
                slices=None, scores=None, examples=None):
        """Send one eval run.

        Returns a dict with gate ('pass', 'suspect' or 'regressed'), verdict, reasons, bug_key and bug_filed,
        returned (keys of fixed bugs that failed again), and over_budget (e.g. ['cost'])."""
        payload = {
            "experiment": experiment,
            "score": float(score),
            "version": version or "",
            "failures": list(failures or []),
            "cost_per_call_usd": cost_per_call_usd,
            "latency_p95_ms": latency_p95_ms,
            "tokens_per_call": tokens_per_call,
        }
        for name, value in (("slices", slices), ("scores", scores), ("examples", examples)):
            if value is not None:
                payload[name] = value
        return self._request(self.endpoint, payload)

    def get_eval_cases(self, experiment):
        """The experiment's eval set: [{"key", "input", "expected"}], including cases promoted from bugs."""
        return self._request(self.base + "/eval-cases?experiment=" + urllib.parse.quote(experiment))["cases"]

    def send_feedback(self, experiment, events):
        """Send production feedback. Each event: {"kind": "negative" | "positive" | "flagged", "input", "output", "version", "trace_id"}."""
        return self._request(self.base + "/feedback-ingest", {"experiment": experiment, "events": list(events)})


# Importers: results from other tools -> one Evalban run. Mirrors actions/evalban-eval/importers.mjs.

def _truthy(v):
    if isinstance(v, bool):
        return v
    return str(v if v is not None else "").strip().lower() in ("true", "1", "yes", "pass", "passed")


def _text(v):
    if v is None:
        return ""
    return v if isinstance(v, str) else json.dumps(v, separators=(",", ":"), ensure_ascii=False)


def _number(v):
    if v is None or v == "" or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f and f not in (float("inf"), float("-inf")) else None


def _round(x):
    r = round(x * 10000) / 10000
    return int(r) if r == int(r) else r


def from_rows(rows):
    if not rows:
        raise ValueError("No results found in the file")
    norm = []
    for i, r in enumerate(rows):
        score = _number(r.get("score"))
        has_passed = r.get("passed") not in (None, "")
        passed = _truthy(r.get("passed")) if has_passed else (score >= 0.5 if score is not None else False)
        key = r.get("id") if r.get("id") is not None else r.get("key") if r.get("key") is not None else "row-%d" % (i + 1)
        sl = r.get("slice") if r.get("slice") is not None else r.get("category") if r.get("category") is not None else ""
        norm.append({"key": _text(key)[:100], "input": _text(r.get("input")), "expected": _text(r.get("expected")),
                     "output": _text(r.get("output")), "passed": passed, "score": score, "slice": _text(sl)})

    def score_of(rs):
        s = [r["score"] for r in rs if r["score"] is not None]
        return _round(sum(s) / len(s) if s else sum(1 for r in rs if r["passed"]) / len(rs))

    out = {
        "score": score_of(norm),
        "failures": [{"input": r["input"], "expected": r["expected"], "output": r["output"]} for r in norm if not r["passed"]][:100],
        "examples": [{"key": r["key"], "passed": r["passed"], "output": r["output"][:2000]} for r in norm[:5000]],
    }
    names = []
    for r in norm:
        if r["slice"] and r["slice"] not in names:
            names.append(r["slice"])
    if names:
        out["slices"] = {n: score_of([r for r in norm if r["slice"] == n]) for n in names[:200]}
    return out


def _from_promptfoo(doc):
    res = doc.get("results") if isinstance(doc, dict) else None
    items = res.get("results") if isinstance(res, dict) else res
    if not isinstance(items, list):
        raise ValueError('This does not look like a promptfoo results file (no "results" list)')
    rows = []
    for i, r in enumerate(items):
        tc = r.get("testCase") or {}
        vars_ = r.get("vars") or tc.get("vars") or {}
        inp = next((vars_[k] for k in ("input", "question", "prompt", "query") if k in vars_), None)
        if inp is None:
            inp = vars_ if vars_ else (r.get("prompt") or {}).get("raw", "")
        if tc.get("description") is not None:
            key = tc["description"]
        elif r.get("testIdx") is not None:
            key = "test-%s%s" % (r["testIdx"], "-p%s" % r["promptIdx"] if r.get("promptIdx") else "")
        else:
            key = "test-%d" % (i + 1)
        grading = r.get("gradingResult") or {}
        meta = tc.get("metadata") or {}
        rows.append({
            "id": key, "input": inp, "expected": vars_.get("expected", vars_.get("answer", "")),
            "output": (r.get("response") or {}).get("output", r.get("output", "")),
            "passed": r.get("success") if r.get("success") is not None else grading.get("pass"),
            "score": r.get("score") if isinstance(r.get("score"), (int, float)) and not isinstance(r.get("score"), bool) else grading.get("score"),
            "slice": meta.get("slice", meta.get("category", vars_.get("slice", vars_.get("category", "")))),
        })
    return from_rows(rows)


def load_results(src, fmt="evalban"):
    """Parse eval results text in one of: evalban, promptfoo, jsonl, csv."""
    if fmt == "evalban":
        return json.loads(src)
    if fmt == "jsonl":
        rows = []
        for i, line in enumerate(l.strip() for l in src.splitlines()):
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                raise ValueError("Line %d is not valid JSON" % (i + 1)) from None
        return from_rows(rows)
    if fmt == "csv":
        reader = csv.reader(io.StringIO(src))
        all_rows = [r for r in reader if any(c.strip() for c in r)]
        if not all_rows:
            return from_rows([])
        head = [h.strip().lower() for h in all_rows[0]]
        return from_rows([{h: (r[i] if i < len(r) else "") for i, h in enumerate(head)} for r in all_rows[1:]])
    if fmt == "promptfoo":
        return _from_promptfoo(json.loads(src))
    raise ValueError('Unknown format "%s". Use evalban, jsonl, csv or promptfoo.' % fmt)


def _main(argv):
    import argparse
    p = argparse.ArgumentParser(prog="python -m evalban", description="Send eval results to Evalban.")
    sub = p.add_subparsers(dest="cmd", required=True)
    lg = sub.add_parser("log", help="Send a results file as one eval run")
    lg.add_argument("--experiment", required=True)
    lg.add_argument("--file", required=True)
    lg.add_argument("--format", default="evalban", choices=["evalban", "promptfoo", "jsonl", "csv"])
    lg.add_argument("--version", default="")
    lg.add_argument("--dry-run", action="store_true", help="Print what would be sent")
    a = p.parse_args(argv)
    with open(a.file, encoding="utf-8") as fh:
        run = load_results(fh.read(), a.format)
    if a.dry_run:
        print(json.dumps(run, indent=2))
        return 0
    client = Evalban(os.environ.get("EVALBAN_API_KEY", ""), os.environ.get("EVALBAN_URL", ""))
    res = client.log_run(a.experiment, run["score"], version=a.version, failures=run.get("failures"),
                         slices=run.get("slices"), scores=run.get("scores"), examples=run.get("examples"))
    print(json.dumps(res))
    return 1 if res.get("gate") == "regressed" else 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))


def _cli():
    """Entry point for the `evalban` command."""
    sys.exit(_main(sys.argv[1:]))
