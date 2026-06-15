"""Backend via headless Claude Code (`claude -p`): a FULL agent.

Unlike a pure text-gen, it inherits Claude Code's agentic loop -- it can use
native tools (Bash, Read, Edit, WebSearch...), MCP and skills. The role prompt
comes in via --append-system-prompt (added to Claude Code's default system).

Auth: by default it uses Claude Code's logged-in session (OAuth) -- it does NOT
need ANTHROPIC_API_KEY. That's why `bare=False` is the default: `--bare` turns off
OAuth (forces an API key) and also skips hooks/CLAUDE.md/MCP. Turn `bare: true` on
only if you use ANTHROPIC_API_KEY and want to isolate from local settings/CLAUDE.md.

Cost: the JSON carries `total_cost_usd` (USD-equivalent cost even on the subscription)
and `usage` with tokens -- these become a Usage. Each call carries the cache overhead
of Claude Code's system prompt, so the floor per call is not zero.
"""
from __future__ import annotations

import asyncio
import json

from ..usage import Completion, Usage
from .base import Backend, BackendError


class ClaudeBackend(Backend):
    def __init__(
        self,
        model: str = "",
        *,
        effort: str | None = None,
        allowed_tools=None,
        disallowed_tools=None,
        permission_mode: str | None = None,
        add_dir=None,
        mcp_config=None,
        bare: bool = False,
        cwd: str | None = None,
        extra_args=None,
        timeout: int = 900,   # effort max + large prompt reasons a lot; 300s used to time out
        executable: str = "claude",
    ):
        self.model = model
        self.effort = effort  # low|medium|high|xhigh|max -> --effort (reasoning)
        self.allowed_tools = list(allowed_tools or [])
        self.disallowed_tools = list(disallowed_tools or [])
        self.permission_mode = permission_mode
        self.add_dir = list(add_dir or [])
        self.mcp_config = list(mcp_config or [])
        self.bare = bare
        self.cwd = cwd
        self.extra_args = list(extra_args or [])
        self.timeout = timeout
        self.executable = executable

    @classmethod
    def from_spec(cls, spec) -> "ClaudeBackend":
        o = spec.options
        return cls(
            model=spec.model,
            effort=o.get("effort"),
            allowed_tools=o.get("allowed_tools") or o.get("tools"),
            disallowed_tools=o.get("disallowed_tools"),
            permission_mode=o.get("permission_mode"),
            add_dir=o.get("add_dir"),
            mcp_config=o.get("mcp_config"),
            bare=o.get("bare", False),
            cwd=o.get("cwd"),
            extra_args=o.get("extra_args"),
            timeout=o.get("timeout", 900),
            executable=o.get("executable", "claude"),
        )

    def build_cmd(self, *, system: str, user: str) -> list[str]:
        """Builds the argv for `claude -p`. Isolated so it's testable without running anything."""
        cmd = [self.executable, "-p", user, "--output-format", "json"]
        if system:
            cmd += ["--append-system-prompt", system]
        if self.model:
            cmd += ["--model", self.model]
        if self.effort:
            cmd += ["--effort", self.effort]
        if self.permission_mode:
            cmd += ["--permission-mode", self.permission_mode]
        # Lists comma-joined into a single token: avoids the variadic parser swallowing flags.
        if self.allowed_tools:
            cmd += ["--allowedTools", ",".join(self.allowed_tools)]
        if self.disallowed_tools:
            cmd += ["--disallowedTools", ",".join(self.disallowed_tools)]
        if self.bare:
            cmd += ["--bare"]
        # Variadic flags (consume up to the next --flag) last, before the extra.
        if self.add_dir:
            cmd += ["--add-dir", *self.add_dir]
        if self.mcp_config:
            cmd += ["--mcp-config", *self.mcp_config]
        cmd += self.extra_args
        return cmd

    @staticmethod
    def _usage_from_json(data: dict, fallback_model: str) -> Usage:
        u = data.get("usage") or {}
        model_used = next(iter(data.get("modelUsage") or {}), fallback_model)
        return Usage(
            backend="claude",
            model=model_used,
            cost_usd=data.get("total_cost_usd"),
            input_tokens=u.get("input_tokens"),
            output_tokens=u.get("output_tokens"),
            cache_read_tokens=u.get("cache_read_input_tokens"),
            cache_write_tokens=u.get("cache_creation_input_tokens"),
        )

    async def generate(self, *, system: str, user: str, temperature: float | None = None) -> Completion:
        # temperature: `claude -p` doesn't expose it. Diversity on the Claude side comes
        # from the role/prompt, not from temp -- ignored on purpose.
        cmd = self.build_cmd(system=system, user=user)
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=self.cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as e:
            raise BackendError(f"executable '{self.executable}' not found in PATH") from e
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=self.timeout)
        except asyncio.TimeoutError as e:
            proc.kill()
            raise BackendError(f"claude -p timed out ({self.timeout}s)") from e

        if proc.returncode != 0:
            msg = err.decode(errors="replace").strip() or f"claude -p failed (rc={proc.returncode})"
            raise BackendError(msg)
        try:
            data = json.loads(out.decode(errors="replace"))
        except json.JSONDecodeError as e:
            raise BackendError(f"non-JSON output from claude -p: {out[:200]!r}") from e
        if data.get("is_error"):
            raise BackendError(str(data.get("result", "claude error")))
        return Completion(text=data.get("result", ""), usage=self._usage_from_json(data, self.model))
