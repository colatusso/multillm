import os

from multillm.env import load_env


def test_loads_and_respects_existing_env(tmp_path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text('# comment\nOPENROUTER_API_KEY="sk-or-xyz"\nexport FOO=bar\nJUNK_NO_EQUALS\n')
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("FOO", "ja-existe")

    assert load_env(f) is True
    assert os.environ["OPENROUTER_API_KEY"] == "sk-or-xyz"  # set from .env
    assert os.environ["FOO"] == "ja-existe"                 # environment wins over .env


def test_missing_file_returns_false(tmp_path):
    assert load_env(tmp_path / "nao-existe.env") is False
