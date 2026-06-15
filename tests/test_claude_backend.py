from multillm.backends.claude import ClaudeBackend
from multillm.config import LLMSpec
from multillm.usage import Usage


def test_build_cmd_minimal():
    b = ClaudeBackend(model="sonnet")
    cmd = b.build_cmd(system="papel", user="oi")
    assert cmd[:5] == ["claude", "-p", "oi", "--output-format", "json"]
    assert cmd[cmd.index("--append-system-prompt") + 1] == "papel"
    assert cmd[cmd.index("--model") + 1] == "sonnet"


def test_build_cmd_tools_permission_and_empty_system():
    b = ClaudeBackend(model="sonnet", allowed_tools=["Read", "Bash"], permission_mode="acceptEdits")
    cmd = b.build_cmd(system="", user="x")
    assert cmd[cmd.index("--allowedTools") + 1] == "Read,Bash"     # comma-joined into one token
    assert cmd[cmd.index("--permission-mode") + 1] == "acceptEdits"
    assert "--append-system-prompt" not in cmd                    # empty system is omitted


def test_build_cmd_variadic_last_and_extra_args():
    b = ClaudeBackend(model="", add_dir=["."], mcp_config=["m.json"], bare=True,
                      extra_args=["--foo", "5"])
    cmd = b.build_cmd(system="s", user="u")
    assert "--bare" in cmd
    assert cmd[-2:] == ["--foo", "5"]                 # extra_args always at the end
    assert "--model" not in cmd                       # empty model is omitted
    assert cmd.index("--add-dir") < cmd.index("--foo")


def test_from_spec_reads_options():
    spec = LLMSpec(name="c", backend="claude", model="opus",
                   options={"tools": ["Read"], "permission_mode": "plan", "bare": True})
    b = ClaudeBackend.from_spec(spec)
    assert b.model == "opus"
    assert b.allowed_tools == ["Read"]
    assert b.permission_mode == "plan"
    assert b.bare is True


def test_usage_from_json():
    data = {
        "result": "ok",
        "total_cost_usd": 0.0277,
        "usage": {"input_tokens": 10, "output_tokens": 62,
                  "cache_read_input_tokens": 17310, "cache_creation_input_tokens": 12804},
        "modelUsage": {"claude-haiku-4-5-20251001": {"costUSD": 0.0277}},
    }
    u = ClaudeBackend._usage_from_json(data, "haiku")
    assert isinstance(u, Usage)
    assert u.cost_usd == 0.0277
    assert u.model == "claude-haiku-4-5-20251001"     # real id comes from modelUsage
    assert (u.input_tokens, u.output_tokens) == (10, 62)
    assert (u.cache_read_tokens, u.cache_write_tokens) == (17310, 12804)
