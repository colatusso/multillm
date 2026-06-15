---
name: multillm
description: Delegates a question to a multi-LLM committee (Mixture-of-Agents) and returns ONE consolidated answer. Several proposers respond in parallel at max reasoning; a stronger model at max effort judges, ranks, absorbs what adds value, fixes errors and writes the final answer. Use ONLY when the question REALLY needs multiple points of view — a hard decision, an ambiguous technical problem, architecture, trade-offs, something where a second/third mind changes the outcome. Triggers (PT/EN) "roda no comitê/multi/multillm" / "run the committee/multi/multillm", "consenso" / "consensus", "segunda opinião dos vários" / "second opinion from several models", "pergunta pros vários" / "ask the several models". Do NOT use for trivial, simple factual or low-risk questions — it's expensive (several calls, multiple strong-model instances at max effort, slow).
metadata:
  tags: multi-llm, mixture-of-agents, consensus, judge, brainstorm
---

# multillm — multi-LLM committee

Delegates the question to a committee (Mixture-of-Agents) and returns the answer
consolidated by the judge. This skill only invokes the project's CLI.

## Setup (once)

Install the package in its environment so it runs from anywhere — no hardcoded paths:

```bash
pip install -e .          # from the project root
```

If you keep it in a specific env (conda/venv), install/run with that env's python.

## How to run

Once installed, run from any directory; pass the question in quotes (escape inner quotes):

```bash
python -m multillm "<QUESTION>"               # mixture (default)
python -m multillm "<QUESTION>" --show-all    # also dump each proposer + reasoning
python -m multillm "<QUESTION>" --mode debate # sequential debate
python -m multillm --show-stats               # leaderboard, no question
```

Need a specific interpreter (conda/venv)? Use it — e.g. `conda run -n <env> python -m
multillm ...`. To avoid retyping that prefix, you may cache it once in a tiny wrapper
at `~/.config/multillm/run` (`#!/usr/bin/env bash` + `conda run -n <env> python -m
multillm "$@"`) and call `~/.config/multillm/run "<QUESTION>"` thereafter.

## Output (important)

- **stdout** = the consolidated **final answer** (standalone, human-friendly). This is what you deliver to the user.
- **stderr** = metadata: mode, live progress (`✓ N/total`), the judge's ranking, observations and per-agent cost.

To separate them: `python -m multillm "..." 2>/tmp/meta.txt` leaves only the answer on stdout.

## How to present it

1. Deliver the final answer (stdout) as the main response.
2. If relevant, summarize the judge's ranking/observations in 1 line (e.g.: "judge: deepseek > opus > qwen; noted that X got Y wrong").
3. If any proposer dropped out (`⚠` in the progress), flag it.

## Notes

- **Heavy:** several proposers + a judge, with multiple strong-model instances at `max effort`. Slow and not cheap — don't use it for trivialities.
- The **claude** backend uses the logged-in session (subscription), not the paid API. The **openrouter** proposers cost money (cents) and require `OPENROUTER_API_KEY` in the project's `.env`.
- Who proposes/judges and the prompts live in `agents.yaml`.
