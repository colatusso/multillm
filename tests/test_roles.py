import pytest

from multillm.agent import Agent
from multillm.roles import debate


async def test_rounds_order_and_chaining(Fake):
    div = Agent("div", Fake(reply="ideas"), "s")
    con = Agent("con", Fake(reply="critique"), "s")
    coh = Agent("coh", Fake(script=["draft1", "draft2"]), "s")
    final, ok = await debate("q", div, con, coh, rounds=2)
    assert final == "draft2"
    assert ok is None
    assert len(div.backend.calls) == 1       # divergent runs once
    assert len(con.backend.calls) == 2       # contrarian per round
    assert len(coh.backend.calls) == 2       # synthesizer per round
    assert "draft1" in con.backend.calls[1]["user"]   # 2nd critique sees the previous draft


async def test_verifier_runs_on_final(Fake):
    mk = lambda r: Agent("x", Fake(reply=r), "s")
    final, ok = await debate("q", mk("i"), mk("c"), mk("d"), rounds=1, verifier=lambda d: d == "d")
    assert final == "d" and ok is True


async def test_invalid_rounds(Fake):
    a = Agent("x", Fake(reply="r"), "s")
    with pytest.raises(ValueError):
        await debate("q", a, a, a, rounds=0)
