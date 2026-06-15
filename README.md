# multillm

**A committee of LLMs that out-reasons any single one of them.**

Several models answer your question in parallel; a strong judge model then ranks
them, throws out what's unproven, absorbs what's good, and writes one final answer
that aims *above* the best individual response. The whole thing is driven by a
single YAML file and runs from the command line — or as a Claude Code skill.

```
                    ┌───────────────────────────-───────────────────┐
   your question ─► │  PROPOSERS  (parallel · max reasoning)        │
                    │  qwen · deepseek · glm · any other model      │
                    │  opus ×2  →  "generator" + "skeptic-killer"   │
                    └───────────────────────┬────-──────────────────┘
                                            │  telegraphic answers
                                            ▼
                    ┌─────────────────────────────────────────-─────┐
                    │  JUDGE  (opus · effort max)                   │
                    │  ranks · kills unproven mechanisms · elevates │
                    └───────────────────────┬──────────────────-────┘
                                            ▼
                        final answer  +  ranking  +  observations
```

## Install & run

```bash
pip install -e .                 # installs multillm + deps (use ".[dev]" to also get pytest)
cp .env.example .env             # add your OPENROUTER_API_KEY
# Claude Code installed & logged in (`claude` on PATH) for the claude backend

python -m multillm "your hard question here"
```

Runs from any directory once installed. The **final answer** goes to **stdout**;
progress, ranking, judge observations and cost go to **stderr**.

## Why

Picking the best of N answers is capped at "the best model" — a fancy router.
**Mixture-of-Agents** instead *synthesizes*: the judge reads every answer, keeps
what survives scrutiny, fixes what's wrong, and can land **past the ceiling** of
any single model. That's the whole point.

Two design tricks make the committee more than the sum of its parts:

- **Different models = free diversity.** qwen, deepseek and glm disagree because
  their training does — they all run the plain `solver` role.
- **Same model, forced angles.** Two `opus` instances would be identical clones,
  so they get *opposite jobs*: `generator` invents from first principles,
  `skeptic_killer` tries to destroy every idea. The judge plays them off.
- **A brutal judge.** The judge rejects any proposal whose mechanism isn't backed
  by evidence, demands real-world viability, and prefers *2 proven ideas over 10
  pretty ones*.

## Example run

Progress streams live, then the consolidated answer (`... > answer.txt` keeps just the answer):

```
▶ MIXTURE mode · proposers: qwen:solver, deepseek:solver, glm:solver,
   claude-opus:generator, claude-opus:skeptic_killer → judge: claude-opus:judge
  ✓ 1/5 · deepseek:solver
  ✓ 2/5 · glm:solver
  ✓ 3/5 · qwen:solver
  ✓ 4/5 · claude-opus:generator
  ✓ 5/5 · claude-opus:skeptic_killer
  ⚖ claude-opus:judge evaluating (effort max)…

<the consolidated answer>

Judge ranking (best→worst): claude-opus:skeptic_killer > deepseek:solver > …
── judge observations ──
deepseek nailed the cost trade-off; qwen over-engineered; glm's claim X was unproven → cut.
Cost per agent:
  …                                          TOTAL  $0.07
```

## The three modes

| mode | what it does | when |
|------|--------------|------|
| **`mixture`** *(default)* | N proposers in parallel → judge **ranks + elevates** into one answer (JSON: `ranking`, `observations`, `answer`) | the real workhorse; hard/ambiguous questions |
| **`council`** | N proposers → a `verifier` (runs a test) or `judge` **picks one** | cheap & grounded when the answer is checkable; ceiling = best individual |
| **`debate`** | `divergent → contrarian → synthesizer`, M rounds | refine a single chain via adversarial critique |

```bash
python -m multillm "your question"               # mixture (default)
python -m multillm "your question" --show-all    # also dump each proposer + reasoning
python -m multillm "your question" --mode debate # sequential debate
echo "your question" | python -m multillm        # or pipe via stdin
```

## Use it from Claude Code (a skill)

multillm ships as a **skill** so Claude Code can delegate to the committee like a
sub-agent. Installed globally at `~/.claude/skills/multillm/` (a copy is versioned
under `skill/` in this repo).

- **Invoke explicitly:** `/multillm <question>`
- **Or just ask** (bilingual triggers): *"run the committee on X"*, *"roda no
  comitê"*, *"get a consensus / segunda opinião dos vários"*. The skill fires,
  runs the CLI, and Claude relays the final answer.

It's a skill, not an MCP server, on purpose: it's a local CLI used by one client,
so a skill (just a `SKILL.md` that shells out) is the simplest thing that works.
MCP would only pay off for a typed tool shared across multiple clients.

> Heads-up: a run is **5 calls, with up to 3 opus instances at max effort** — slow
> and not cheap. Use it when a second/third mind actually changes the outcome, not
> for trivia. The skill's description tells Claude exactly that.

## Configuration — everything lives in `agents.yaml`

Two reusable blocks plus the team. An agent's system prompt is composed:
`base_prompt` (the LLM's *voice*) **+** the role's `prompt` (its *function*).

```yaml
llms:                          # the "voice" + backend of each model
  claude-opus:
    backend: claude            # uses your logged-in Claude Code session (no API key)
    model: claude-opus-4-8     # full id pins the version
    effort: xhigh              # reasoning: low|medium|high|xhigh|max
  qwen:
    backend: openrouter
    model: qwen/qwen3.7-max
    reasoning: { effort: high }

roles:                         # the "function" — solver, judge, generator, skeptic_killer…
  solver:   { prompt: "…telegraphic, dense, you write for the JUDGE not a human…" }
  judge:    { prompt: "…brutal: reject unproven mechanisms, rank, elevate…" }

team:                          # who proposes / who judges (mixture mode)
  proposers:
    - { llm: qwen,        role: solver }
    - { llm: deepseek,    role: solver }
    - { llm: glm,         role: solver }
    - { llm: claude-opus, role: generator }       # opus angle A: invents
    - { llm: claude-opus, role: skeptic_killer }  # opus angle B: kills
  synthesizer: { llm: claude-opus, role: judge }
```

Add a model, swap the judge, retune a prompt, change who's on the team — all here,
no code. Unknown LLM fields (`tools`, `permission_mode`, `mcp_config`, `max_tokens`…)
are passed straight to the backend.

## Output, tracking & cost

- **stdout** = the standalone final answer (it never mentions the candidates).
- **stderr** = mode header, live `✓ N/total` progress, judge ranking, observations,
  per-agent cost. A proposer that times out or returns empty shows as `⚠` and is
  dropped — the committee keeps going instead of failing.
- **Tracking:** every run records who the judge ranked where, in `multillm_stats.json`.

```bash
python -m multillm --show-stats     # leaderboard by win %
python -m multillm "…" --no-track   # skip recording this run
```

```
Tracking (best by win %):
  deepseek/deepseek-v4-pro   60%  (3/5 wins)  borda=11
  claude-opus-4-8            40%  (2/5 wins)  borda=9
  …
```

## Backends

| backend      | what it is | tools / MCP | cost |
|--------------|-----------|-------------|------|
| `claude`     | Claude Code headless (`claude -p`) — a **full agent** | yes (native tools, MCP, skills) | `total_cost_usd` from the JSON |
| `openrouter` | OpenAI-compatible HTTP (Qwen, DeepSeek, GLM…) | no (text-gen) | `usage.cost` |

The **claude backend uses your logged-in session (OAuth)** — no `ANTHROPIC_API_KEY`,
billed to your subscription, not the paid API. Reasoning maxes via `effort` (claude)
or `reasoning: {effort: high}` (openrouter); `message.reasoning` is captured and
shown under `--show-all`.

## Python API

```python
import asyncio
from multillm import load_config, build_team, mixture, format_cost_report

cfg  = load_config("agents.yaml")
team = build_team(cfg)
result = asyncio.run(mixture("your question", team.proposers, team.synthesizer))

print(result.final)              # the consolidated answer
print(result.ranking)            # indices best→worst
print(result.observations)       # the judge's notes
print(format_cost_report(team.agents))
```

Or assemble agents by hand — `build_agent(cfg.llm("qwen"), cfg.role("solver"))` —
and call `council(...)` / `debate(...)` directly.

## Project layout

```
multillm/
  __main__.py   CLI entrypoint (pass the question)
  config.py     YAML → typed specs (LLMSpec / RoleSpec / Config)
  prompts.py    compose_system — LLM voice + role
  agent.py      Agent (backend + system) + build_agent; accrues cost in .usages
  team.py       build_team — proposers + synthesizer from the `team` block
  backends/     base (contract) · claude (claude -p) · openrouter · registry
  council.py    gather_all + council (verifier/judge selector); Candidate
  mixture.py    Mixture-of-Agents: propose → judge (JSON) → elevate
  roles.py      debate (divergent → contrarian → synthesizer)
  usage.py      Usage / Completion (text + cost + reasoning)
  costs.py      per-agent cost report
  tracking.py   win-rate tracking (persisted JSON)
  env.py        load_env (.env, zero dependency)
agents.yaml     LLMs + roles + team/debate — edit everything here
```

## Extending

- **New backend** (Gemini, Ollama…): implement `Backend`, add one line to
  `backends/__init__.py`. Nothing else changes (Open/Closed).
- **Verifier for any domain:** `council(..., verifier=fn)` with any
  `Callable[[str], bool]` — compile, lint, validate a schema, run a test.
- **Nest:** run `debate` K times and `council` to pick among the runs.

## Tests & safety

```bash
pytest        # 47 tests — fake backend, no network, no spend
```

> ⚠️ `run_python` (the example verifier) executes LLM-generated code locally. Use a
> sandboxed container in production.
