"""Render a run trace (JSONL) into a short readable timeline.

    python -m teamscale.trace_summary experiments/traces/<run>.jsonl

The goal is the one from the job description: a researcher should be able to
read what a large team did -- and where it went wrong -- in minutes.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def summarize_trace(path: str) -> str:
    events = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    start = next((e for e in events if e["event"] == "run_start"), {})
    end = next((e for e in reversed(events) if e["event"] == "run_end"), {})
    counts = Counter(e["event"] for e in events)
    agent_tokens = defaultdict(int)
    agent_tasks = Counter()
    tokens_by_event = defaultdict(int)
    failures, retries = [], 0
    for e in events:
        # Token accounting: assign = primary attempt, retry = extra attempt,
        # verify = verifier pass. subtask_done echoes the job total and is
        # excluded, or every token would be counted twice.
        if e["event"] in ("assign", "retry", "verify", "duplicate"):
            tokens_by_event[e["event"]] += e.get("tokens", 0)
        if e["event"] == "assign":
            agent_tokens[e["agent"]] += e.get("tokens", 0)
            agent_tasks[e["agent"]] += 1
        if e["event"] == "retry":
            agent_tokens[e["agent"]] += e.get("tokens", 0)
        if e["event"] == "subtask_done" and not e.get("correct", True):
            failures.append(e["subtask"] + (f" ({e['note']})" if e.get("note") else ""))
        if e["event"] == "retry":
            retries += 1
    busiest = max(agent_tokens.items(), key=lambda kv: kv[1], default=(None, 0))
    lines = [
        f"trace: {Path(path).name}",
        f"run: {start.get('family', '?')} / {start.get('strategy', '?')} / "
        f"team of {start.get('team_size', '?')} / budget {start.get('budget', '?')}",
        f"outcome: success={end.get('success')} fraction_correct={end.get('fraction_correct')} "
        f"tokens={end.get('total_tokens')} wall_time={end.get('wall_time')}",
        f"activity: {counts.get('assign', 0)} assignments, {counts.get('duplicate', 0)} duplicates, "
        f"{counts.get('verify', 0)} verifications, {retries} retries",
        f"busiest agent: agent {busiest[0]} -- {agent_tasks[busiest[0]]} subtasks, {busiest[1]} tokens",
        "failures: " + (", ".join(failures) if failures else "none"),
        "where the tokens went: " + ", ".join(
            f"{k} {v}" for k, v in sorted(tokens_by_event.items(), key=lambda kv: -kv[1]) if v),
        "",
        "timeline:",
    ]
    for e in events:
        if e["event"] in ("run_start", "run_end"):
            continue
        bits = [f"  t={e['t']:>7}  {e['event']:<14}"]
        if "subtask" in e:
            bits.append(str(e["subtask"]))
        if "agent" in e:
            bits.append(f"agent {e['agent']}")
        if e.get("tokens"):
            bits.append(f"{e['tokens']} tok")
        if "correct" in e:
            bits.append("ok" if e["correct"] else "FAILED")
        if e.get("note"):
            bits.append(e["note"])
        lines.append(" ".join(bits))
    return "\n".join(lines)


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: python -m teamscale.trace_summary <trace.jsonl>")
        return 2
    print(summarize_trace(args[0]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
