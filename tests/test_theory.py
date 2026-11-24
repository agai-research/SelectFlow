import itertools
import math

import numpy as np

from selectflow import synth
from selectflow.certification import benjamini_hochberg, conformal_pvalues
from selectflow.compose import compose
from selectflow.derivation import check, forward_chain
from selectflow.model import Context, Descriptor, Federation, ModelHost, Params, QoS, Server, Tool, Workflow
from selectflow.selection import Featurizer, Observation, Quality, SelectionModel, fit


def tiny(n=6, seed=0, preamble=0, shared=False):
    rng = np.random.default_rng(seed)
    servers = {f"s{i}": Server(f"s{i}", "O", preamble) for i in range(n)}
    tools = {}
    for i in range(n):
        tools[f"t{i}"] = Tool(f"t{i}", "s0" if shared else f"s{i}", "O", "x", {"r": 0.9}, int(rng.integers(200, 900)),
                             QoS(1, 0), inp=set(), adequacy={"r": float(rng.uniform(0, 1))},
                             psi=float(rng.uniform(0, 0.3)), attrs={"sim": float(rng.uniform(0, 1)),
                             "assertive": float(rng.integers(0, 2))})
    D = {"v": Descriptor("v", "O", "p", set(), {"out"}, {"r"})}
    fed = Federation(D, tools, servers, {"h": ModelHost("h", "O", 2500)}, set(), {"out": set()}, set())
    fed.certified = {"fwd": set(tools)}
    m = SelectionModel(np.array([-0.3, 1.5, 0.2, 0.3, 1.0, 0.2]), 0.5, Featurizer(fed))
    return fed, m


def test_prop2_diluted_prefix_is_optimal():
    for seed in range(15):
        fed, m = tiny(seed=seed)
        q = Quality(fed, Params(), m)
        best = max((q.Q_r(list(K), "r", "v"), K) for k in range(0, 7) for K in itertools.combinations(fed.tools, k))
        order = sorted(fed.tools, key=lambda t: -q.diluted(t, "r", "v"))
        prefix = max(q.Q_r(order[:k], "r", "v") for k in range(0, 7))
        assert abs(prefix - best[0]) < 1e-9


def test_prop2b_marginal_condition():
    fed, m = tiny(seed=3)
    q = Quality(fed, Params(), m)
    K = ["t0", "t1"]
    for t in ["t2", "t3", "t4"]:
        a = m.attraction(t, "r", "v")
        delta = fed.tools[t].cost / 1000
        cond = q.w(t, "r") > q.Q_r(K, "r", "v") * (1 + m.gamma * delta / a)
        assert cond == (q.Q_r(K + [t], "r", "v") > q.Q_r(K, "r", "v"))


def test_milp_is_exact_against_brute_force():
    for seed in range(8):
        fed, m = tiny(n=6, seed=seed, preamble=120, shared=(seed % 2 == 0))
        params = Params(k_max=4, N=6, n_min=1, tau_mu=0.5)
        W = Workflow(["v"], {})
        res = compose(fed, W, set(), {"h": m}, params, Context(set()))
        q = Quality(fed, params, m)
        feas = [K for k in range(1, 5) for K in itertools.combinations(fed.tools, k) if fed.load(K) <= 2500]
        best = max(q.Q(list(K), "v") for K in feas)
        assert abs(res.plan["v"].Q - best) < 1e-6


def test_fit_recovers_parameters():
    fed, models = synth.generate(synth.SynthConfig(seed=1, n_procedures=20))
    agent = synth.SimulatedAgent(models["hP"], seed=2)
    rng = np.random.default_rng(3)
    caps = [(r, v) for v, d in fed.descriptors.items() for r in d.R]
    obs = []
    for _ in range(12000):   # gamma and the intercept are only separated when the load varies widely
        r, v = caps[int(rng.integers(len(caps)))]
        pool = list(fed.tools)
        K = list(rng.choice(pool, int(rng.integers(2, 20)), replace=False))
        obs.append(Observation(K, r, v, agent.select(K, r, v)))
    m, _ = fit(obs, models["hP"].featurizer)
    assert np.allclose(m.beta, synth.TRUE_BETA, atol=0.35)
    assert abs(m.gamma - synth.TRUE_GAMMA) < 0.2


def test_conformal_selection_controls_fdr():
    rng = np.random.default_rng(0)
    eps, fdps = 0.1, []
    for _ in range(300):
        n = 400
        unsafe = rng.random(n) < 0.3
        score = np.where(unsafe, rng.normal(0.3, 0.2, n), rng.normal(0.7, 0.2, n))
        aud = rng.random(n) < 0.3
        p = conformal_pvalues(score[~aud], score[aud], unsafe[aud])
        sel = benjamini_hochberg(p, eps)
        k = sel.sum()
        fdps.append((unsafe[~aud][sel].sum() / k) if k else 0.0)
    assert np.mean(fdps) <= eps + 0.01


def test_forward_chain_is_complete():
    rng = np.random.default_rng(5)
    for trial in range(60):
        n = 5
        D = {}
        for i in range(n):
            In = set(rng.choice([f"a{j}" for j in range(4)], int(rng.integers(0, 2)), replace=False))
            Out = set(rng.choice([f"a{j}" for j in range(4)], 1))
            comp = {f"d{int(rng.integers(n))}"} if rng.random() < 0.15 else set()
            D[f"d{i}"] = Descriptor(f"d{i}", "O", "p", In, Out, {"r"}, comp=comp - {f"d{i}"})
        G = {"a3"}
        exists = False
        for k in range(1, n + 1):
            for S in itertools.combinations(D, k):
                for order in itertools.permutations(S):
                    rep, _ = check(list(order), G, set(), D, 0.5)
                    if not (rep.missing or rep.targets or rep.companions):
                        exists = True; break
                if exists: break
            if exists: break
        order, U = forward_chain(D, G, set(), 0.5)
        assert (order is not None) == exists
        if order is not None:
            rep, _ = check(order, G, set(), D, 0.5)
            assert rep.empty()


def test_milp_exact_with_negative_load_effect():
    for seed in range(6):
        fed, m = tiny(n=6, seed=seed, preamble=120, shared=(seed % 2 == 0))
        m.gamma = -0.25                      # more definitions, fewer abstentions (budget 2.5k tokens)
        params = Params(k_max=4, N=6, n_min=1, tau_mu=0.5)
        res = compose(fed, Workflow(["v"], {}), set(), {"h": m}, params, Context(set()))
        q = Quality(fed, params, m)
        feas = [K for k in range(1, 5) for K in itertools.combinations(fed.tools, k) if fed.load(K) <= 2500]
        assert abs(res.plan["v"].Q - max(q.Q(list(K), "v") for K in feas)) < 1e-6
