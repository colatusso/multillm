import pytest

from multillm.agent import Agent
from multillm.council import council


async def test_no_selector_returns_first(Fake):
    ps = [Agent("a", Fake(reply="r1"), "s"), Agent("b", Fake(reply="r2"), "s")]
    ans, how = await council("q", ps)
    assert ans == "r1" and how == "no-selector"


async def test_verifier_picks_passing(Fake):
    ps = [Agent("a", Fake(reply="ruim"), "s"), Agent("b", Fake(reply="BOM"), "s")]
    ans, how = await council("q", ps, verifier=lambda c: c == "BOM")
    assert ans == "BOM" and how == "verifier"


async def test_verifier_fallback_when_none_pass(Fake):
    ps = [Agent("a", Fake(reply="x"), "s")]
    ans, how = await council("q", ps, verifier=lambda c: False)
    assert ans == "x" and how == "verifier-fallback"


async def test_judge_returns_index(Fake):
    ps = [Agent("a", Fake(reply="r0"), "s"), Agent("b", Fake(reply="r1"), "s")]
    judge = Agent("j", Fake(reply="the best is 1"), "s")
    ans, how = await council("q", ps, judge=judge)
    assert ans == "r1" and how == "judge->1"


async def test_judge_clamps_out_of_range(Fake):
    ps = [Agent("a", Fake(reply="r0"), "s")]
    judge = Agent("j", Fake(reply="9"), "s")
    ans, how = await council("q", ps, judge=judge)
    assert ans == "r0" and how == "judge->0"


async def test_all_proposers_fail_raises(Failing):
    ps = [Agent("a", Failing(), "s")]
    with pytest.raises(RuntimeError):
        await council("q", ps)


async def test_empty_proposers_raises(Fake):
    with pytest.raises(ValueError):
        await council("q", [])


def test_candidate_ok_property():
    from multillm.council import Candidate
    assert Candidate("a", "m", "texto").ok
    assert not Candidate("a", "m", "   ").ok               # empty text
    assert not Candidate("a", "m", "x", error="boom").ok   # with error


async def test_gather_all_keeps_failures_without_dropping(Fake, Failing):
    from multillm.council import gather_all
    ps = [Agent("ok", Fake(reply="boa"), "s"), Agent("ruim", Failing(), "s")]
    allc = await gather_all("q", ps)
    by = {c.agent: c for c in allc}
    assert len(allc) == 2                       # nobody disappears
    assert by["ok"].ok is True
    assert by["ruim"].ok is False and by["ruim"].error is not None
