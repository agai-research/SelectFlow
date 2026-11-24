from selectflow import scenario_greece as G
from selectflow.compose import compose
from selectflow.derivation import ScriptedGenerator, derive
from selectflow.flows import admissible_hosts, prohibitions
from selectflow.repair import solve_request


def setup():
    fed = G.build()
    return fed, G.models(fed)


def test_derivation_repairs_missing_attestation():
    fed, _ = setup()
    res = derive(G.REQUEST, G.CTX, fed, G.PARAMS, G.interpreter, ScriptedGenerator([["v1", "v3", "v4"]]))
    orders = [sorted(w.order) for w in res.workflows]
    assert ["v1", "v2", "v3", "v4"] in orders
    assert res.log["rejected_by_critics"] == 1
    w = [w for w in res.workflows if sorted(w.order) == ["v1", "v2", "v3", "v4"]][0]
    assert w.ho[("v2", "v4")] == {"site_attestation"}
    assert w.ho[("v1", "v2")] == {"tax_registration"}


def test_hosts_and_prohibitions():
    fed, _ = setup()
    w = derive(G.REQUEST, G.CTX, fed, G.PARAMS, G.interpreter, None).workflows[0]
    assert admissible_hosts(fed, w, "v1") == ["hT"]          # fiscal data cannot reach the contractor
    assert "hP" in admissible_hosts(fed, w, "v4")
    P = prohibitions(fed, w, "v4", list(fed.tools), ["hP"])
    assert "muni.upload" in P.F                               # not certified as non-forwarding
    assert "tax.debt.status" in P.forbidden_under("hP")       # returns FIN, outside the contract


def test_end_to_end_outcome():
    fed, M = setup()
    nd, best, info = solve_request(G.REQUEST, G.CTX, fed, G.PARAMS, M, G.interpreter,
                                   ScriptedGenerator(G.CANDIDATES))
    assert best is not None
    assert best.gap == {"v3"}                                 # no tool submits the fire application
    assert best.plan["v1"].host == "hT"
    assert best.plan["v2"].host == "hP" and best.plan["v4"].host == "hP"
    K4 = set(best.plan["v4"].K)
    assert "notify.new.submit" in K4 and "notify.change.submit" not in K4
    assert info["ledger"] and info["ledger"][0]["descriptor"] == "v3"
    assert best.dur == 1 + 15 + 1                             # critical path, v2 and v3 in parallel
