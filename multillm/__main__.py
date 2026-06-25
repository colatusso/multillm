"""CLI: pass the question and run multi-LLM.

  python -m multillm "your question"                # mixture: proposers + judge (ELEVATES)
  python -m multillm "your question" --show-all     # also shows each model's answer
  python -m multillm "your question" --mode debate  # debate: divergent -> contrarian -> synthesizer
  python -m multillm --show-stats                   # tracking: who the judge picks most
  echo "your question" | python -m multillm         # also reads from stdin
  python -m multillm < prompt.txt                   # big prompt: read the whole file from stdin

Ad-hoc proposers (ignore the yaml `team`, bench OpenRouter ids by comma) -- the
judge stays the yaml synthesizer unless you pass --judge:

  python -m multillm "q" -m qwen/qwen3-max,deepseek/deepseek-r1,z-ai/glm-5.1
  python -m multillm "q" -m "qwen/qwen3-max@generator,deepseek/deepseek-r1"  # role via @

The FINAL answer goes to stdout; candidates, ranking, cost and tracking go to stderr
(so you can redirect just the answer: `python -m multillm "x" > out.txt`). It calls the
real models. The team (who proposes / reviews) comes from agents.yaml.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from .agent import build_agent
from .config import Config, LLMSpec, load_config
from .costs import format_cost_report
from .env import load_env
from .mixture import MixtureResult, mixture
from .roles import debate
from .team import Team, agent_from_ref, build_team
from .tracking import format_stats, record_result

DEFAULT_CFG = str(Path(__file__).resolve().parent.parent / "agents.yaml")


def _split_spec(spec_str: str, default_role: str) -> tuple[str, str]:
    """'model[@role]' -> (model_id, role_name).

    The role separator is '@', not ':', so OpenRouter variant ids like
    'deepseek/deepseek-r1:free' or '…:nitro' stay intact. Empty role -> default.
    """
    model_id, _, role = spec_str.partition("@")
    return model_id.strip(), (role.strip() or default_role)


def _adhoc_agent(cfg: Config, spec_str: str, default_role: str, effort: str):
    """Builds an OpenRouter agent straight from a CLI 'id[@role]' (no yaml `llms` entry).

    The model id is used as-is; the role prompt still comes from the yaml `roles`
    block (so the brainstorm prompts stay DRY). Reasoning effort is applied, and the
    cheapest provider is preferred ({"sort": "price"}) so benchmarks aren't overpriced
    by random provider routing.
    An unknown role raises KeyError (handled in main as a clean error).
    """
    model_id, role_name = _split_spec(spec_str, default_role)
    llm = LLMSpec(name=model_id, backend="openrouter", model=model_id,
                  options={"reasoning": effort, "provider": {"sort": "price"}})
    return build_agent(llm, cfg.role(role_name))


def _adhoc_team(cfg: Config, args) -> Team:
    """Team from --models (ignores the yaml `team`); judge = --judge or yaml synthesizer."""
    proposers = [_adhoc_agent(cfg, s, args.role, args.effort)
                 for s in args.models.split(",") if s.strip()]
    if args.judge:
        synth = _adhoc_agent(cfg, args.judge, "judge", args.effort)
    else:
        ref = (cfg.raw.get("team") or {}).get("synthesizer")
        synth = agent_from_ref(cfg, ref) if ref else None
    return Team(proposers=proposers, synthesizer=synth)


_W = 72


def _trunc(text: str, n: int) -> str:
    text = text.strip()
    return text if len(text) <= n else text[:n].rstrip() + f"\n… [+{len(text) - n} chars cut]"


def _indent(text: str, prefix: str = "  │ ") -> str:
    return "\n".join(prefix + ln for ln in text.splitlines())


def _format_candidates(result: MixtureResult) -> str:
    n_ok, n_drop = len(result.candidates), len(result.dropped)
    head = f"  PROPOSERS · {n_ok} with answer" + (f" · {n_drop} with NO answer" if n_drop else "")
    out = ["", "═" * _W, head, "═" * _W]
    for i, c in enumerate(result.candidates):
        out += ["", "┌" + "─" * (_W - 1), f"│ [{i}]  {c.agent}   ·   {c.model}", "└" + "─" * (_W - 1)]
        if c.reasoning:
            r = c.reasoning.strip()
            out += [f"  ╶ reasoning ({len(r)} chars):", _indent(_trunc(r, 1000)), "  ╶ answer:"]
        out.append(c.text.strip())
    for c in result.dropped:
        reason = c.error or "empty answer (reasoning may have blown past max_tokens)"
        out += ["", "┌" + "─" * (_W - 1), f"│ [—]  {c.agent}   ·   {c.model}   ⚠ NO ANSWER",
                "└" + "─" * (_W - 1), f"  reason: {reason}"]
    out += ["", "═" * _W]
    return "\n".join(out)


def _format_ranking(result: MixtureResult) -> str:
    ranked = result.ranked()
    if not ranked:
        return "Judge ranking: (no ranking returned)"
    return "Judge ranking (best->worst): " + "  >  ".join(c.agent for c in ranked)


async def _run(args, question: str) -> None:
    cfg = load_config(args.config)

    if args.mode == "mixture":
        team = _adhoc_team(cfg, args) if args.models else build_team(cfg)
        if not team.proposers:
            raise SystemExit("mixture mode needs proposers (yaml team block or --models)")
        if team.synthesizer is None:
            raise SystemExit("mixture mode needs a judge ('synthesizer' in the yaml team or --judge)")
        src = "--models" if args.models else "team block"
        props = ", ".join(p.name for p in team.proposers)
        print(f"▶ MIXTURE mode ({src}) · proposers: {props} → judge: {team.synthesizer.name}",
              file=sys.stderr, flush=True)

        def _prog(done, total, c):
            mark = "✓" if c.ok else "⚠"
            extra = "" if c.ok else f" — {c.error or 'empty'}"
            print(f"  {mark} {done}/{total} · {c.agent}{extra}", file=sys.stderr, flush=True)

        def _judging():
            print(f"  ⚖ {team.synthesizer.name} evaluating…", file=sys.stderr, flush=True)

        result = await mixture(question, team.proposers, team.synthesizer,
                               on_proposer=_prog, on_judge=_judging)

        if args.show_all:
            print(_format_candidates(result), file=sys.stderr)
        elif result.dropped:  # even without --show-all, don't let it vanish silently
            d = ", ".join(f"{c.agent} ({c.error or 'empty'})" for c in result.dropped)
            print(f"⚠ {len(result.dropped)} proposer(s) with no answer: {d}", file=sys.stderr)
        print(result.final)  # stdout: the final answer

        if not args.no_track:
            record_result(result, args.stats_file)
        print("\n" + _format_ranking(result), file=sys.stderr)
        if result.observations:
            print(f"\n── judge observations ──\n{result.observations}", file=sys.stderr)
        print(format_cost_report(team.agents), file=sys.stderr)

    elif args.mode == "debate":
        d = cfg.raw.get("debate")
        if not d:
            raise SystemExit("debate mode requires a 'debate' block in the yaml")
        div = agent_from_ref(cfg, d["divergent"])
        con = agent_from_ref(cfg, d["contrarian"])
        coh = agent_from_ref(cfg, d["synthesizer"])
        rounds = d.get("rounds", 2)
        print(f"▶ DEBATE mode (debate block) · {div.name} → {con.name} → {coh.name} · {rounds} rounds",
              file=sys.stderr)
        final, _ = await debate(question, div, con, coh, rounds=rounds)
        print(final)
        print("\n" + format_cost_report([div, con, coh]), file=sys.stderr)


def main() -> None:
    load_env()  # load .env (OPENROUTER_API_KEY etc.) before building the agents
    ap = argparse.ArgumentParser(description="multi-LLM: pass the question and run.")
    ap.add_argument("question", nargs="?", help="the question (or via stdin)")
    ap.add_argument("--config", default=DEFAULT_CFG, help="path to agents.yaml")
    ap.add_argument("--mode", choices=["mixture", "debate"], default="mixture",
                    help="mixture (proposers + judge, default) or debate")
    ap.add_argument("-m", "--models", default="",
                    help="comma-separated OpenRouter ids as ad-hoc proposers (ignores the yaml "
                         "team); per-model role via 'id@role', else --role. e.g. "
                         "'qwen/qwen3-max,deepseek/deepseek-r1@generator'")
    ap.add_argument("--role", default="solver",
                    help="default role for --models entries without '@role' (default: solver)")
    ap.add_argument("--judge", default="",
                    help="ad-hoc judge 'id[@role]' for --models runs (default: the yaml synthesizer)")
    ap.add_argument("--effort", default="high",
                    help="reasoning effort for --models / --judge agents (default: high)")
    ap.add_argument("--show-all", action="store_true",
                    help="show each proposer's answer (and reasoning) on stderr")
    ap.add_argument("--stats-file", default="multillm_stats.json", help="tracking file")
    ap.add_argument("--no-track", action="store_true", help="don't record tracking on this run")
    ap.add_argument("--show-stats", action="store_true",
                    help="print the accumulated tracking and exit (doesn't run a question)")
    args = ap.parse_args()

    if args.show_stats:
        print(format_stats(args.stats_file))
        return

    question = args.question if args.question else sys.stdin.read().strip()
    if not question:
        ap.error("missing question (pass it as an argument or via stdin)")
    try:
        asyncio.run(_run(args, question))
    except (RuntimeError, KeyError) as e:  # BackendError(RuntimeError) + unknown role/llm (KeyError)
        print(f"\n✖ committee failed: {e}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
