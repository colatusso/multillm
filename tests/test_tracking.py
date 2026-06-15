from multillm.council import Candidate
from multillm.mixture import MixtureResult
from multillm.tracking import format_stats, load_stats, record_result


def _result(ranking):
    cands = [
        Candidate(agent="qwen:solver", model="qwen", text="a"),
        Candidate(agent="deepseek:solver", model="deepseek", text="b"),
        Candidate(agent="sonnet:solver", model="sonnet", text="c"),
    ]
    return MixtureResult(final="f", ranking=ranking, candidates=cands)


def test_record_accumulates_wins_and_borda(tmp_path):
    path = tmp_path / "stats.json"
    record_result(_result([1, 0, 2]), path)   # deepseek 1st, qwen 2nd, sonnet 3rd
    record_result(_result([1, 2, 0]), path)   # deepseek 1st again

    stats = load_stats(path)
    assert stats["deepseek"] == {"runs": 2, "wins": 2, "borda": 6}   # n=3: 1st=3 pts, x2
    assert stats["qwen"]["runs"] == 2 and stats["qwen"]["wins"] == 0
    assert stats["qwen"]["borda"] == 3          # 2nd(=2) + 3rd(=1)


def test_record_without_ranking_counts_runs_only(tmp_path):
    path = tmp_path / "stats.json"
    record_result(_result(None), path)
    stats = load_stats(path)
    assert stats["qwen"] == {"runs": 1, "wins": 0, "borda": 0}


def test_format_stats_empty_and_populated(tmp_path):
    assert "(no data yet)" in format_stats(tmp_path / "nope.json")
    path = tmp_path / "stats.json"
    record_result(_result([1, 0, 2]), path)
    out = format_stats(path)
    assert "deepseek" in out and "wins" in out
