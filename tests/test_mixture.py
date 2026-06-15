import pytest

from multillm.agent import Agent
from multillm.mixture import _clean_ranking, _parse_judge, mixture


async def test_judge_json_ranking_obs_and_answer(Fake):
    proposers = [Agent("a", Fake(reply="resp A"), "s"), Agent("b", Fake(reply="resp B"), "s")]
    judge_json = '{"ranking": [1, 0], "observations": "B mais completa", "answer": "FINAL elevado"}'
    synth = Agent("opus", Fake(reply=judge_json), "s")

    result = await mixture("question X", proposers, synth)

    assert result.final == "FINAL elevado"
    assert result.ranking == [1, 0]
    assert result.observations == "B mais completa"
    assert [c.text for c in result.candidates] == ["resp A", "resp B"]
    u = synth.backend.calls[0]["user"]
    assert "resp A" in u and "resp B" in u and "question X" in u   # judge sees everything
    assert result.winner().agent == "b"


async def test_invalid_json_falls_back_to_text(Fake):
    proposers = [Agent("a", Fake(reply="x"), "s")]
    synth = Agent("opus", Fake(reply="isso nao e json"), "s")
    result = await mixture("q", proposers, synth)
    assert result.ranking is None
    assert result.observations == ""
    assert result.final == "isso nao e json"          # raw text becomes the answer


async def test_empty_proposer_goes_to_dropped_not_silence(Fake):
    proposers = [Agent("boa", Fake(reply="valid answer"), "s"),
                 Agent("vazio", Fake(reply="   "), "s")]   # empty content (e.g. reasoning overflowed)
    synth = Agent("opus", Fake(reply='{"ranking":[0],"observations":"","answer":"final"}'), "s")
    result = await mixture("q", proposers, synth)
    assert [c.agent for c in result.candidates] == ["boa"]    # only the non-empty one enters the synthesis
    assert [c.agent for c in result.dropped] == ["vazio"]     # the empty one does NOT disappear
    assert result.dropped[0].ok is False
    assert "valid answer" in synth.backend.calls[0]["user"]


async def test_empty_proposers_raises(Fake):
    synth = Agent("opus", Fake(reply="{}"), "s")
    with pytest.raises(ValueError):
        await mixture("q", [], synth)


def test_parse_judge_plain_json():
    r, o, resp = _parse_judge('{"ranking":[2,0,1],"observations":"ok","answer":"y"}', 3)
    assert r == [2, 0, 1] and o == "ok" and resp == "y"


def test_parse_judge_with_fences_and_surrounding_text():
    raw = 'segue:\n```json\n{"ranking":[0],"observations":"o","answer":"r"}\n```\nfim'
    r, o, resp = _parse_judge(raw, 1)
    assert r == [0] and o == "o" and resp == "r"


def test_clean_ranking_filters_invalid_and_dups():
    assert _clean_ranking([2, 0, 5, 2, 1], 3) == [2, 0, 1]   # drops 5 (out of range) and the repeated 2
    assert _clean_ranking("nope", 3) is None
    assert _clean_ranking([], 3) is None


async def test_progress_callbacks_fire(Fake):
    proposers = [Agent("a", Fake(reply="x"), "s"), Agent("b", Fake(reply="y"), "s")]
    synth = Agent("opus", Fake(reply='{"ranking":[0,1],"observations":"","answer":"f"}'), "s")
    seen, judged = [], []
    await mixture("q", proposers, synth,
                  on_proposer=lambda done, total, c: seen.append((done, total, c.agent)),
                  on_judge=lambda: judged.append(True))
    assert [s[0] for s in seen] == [1, 2]          # counter 1,2
    assert all(s[1] == 2 for s in seen)            # total = 2
    assert {s[2] for s in seen} == {"a", "b"}      # both notified
    assert judged == [True]                         # judge signaled once, after the proposers
