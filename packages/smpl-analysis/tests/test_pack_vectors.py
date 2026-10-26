"""cohesion / coverage / probe on synthetic vectors with known answers (vault-16dd.5).

No models: every vector is built here, so each test has a positive and a negative control.
"""

from __future__ import annotations

import numpy as np
import pytest
from smpl_analysis import cohesion as C
from smpl_analysis import coverage as CV
from smpl_analysis import probe as P
from smpl_analysis import vectors as V

DIM = 32


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def _cluster(center, n, spread, rng):
    return np.stack([_unit(center + spread * rng.standard_normal(DIM)) for _ in range(n)])


@pytest.fixture()
def rng():
    return np.random.default_rng(7)


# --- cohesion -------------------------------------------------------------------------------

def test_cohesion_flags_the_planted_outlier_only(rng):
    a = _unit(rng.standard_normal(DIM))
    pack = _cluster(a, 10, 0.15, rng)
    odd = _unit(rng.standard_normal(DIM))  # a different direction entirely
    res = C.cohesion(np.vstack([pack, odd]))
    assert res["outlier"].tolist() == [False] * 10 + [True]
    assert res["outlier_z"][-1] > 3.5
    # negative control: the same pack without the stranger flags nothing
    assert not C.cohesion(pack)["outlier"].any()


def test_pairwise_distance_tracks_spread(rng):
    a = _unit(rng.standard_normal(DIM))
    tight = C.mean_pairwise_distance(_cluster(a, 12, 0.05, rng))
    loose = C.mean_pairwise_distance(_cluster(a, 12, 0.6, rng))
    assert 0 <= tight < 0.1 < loose
    assert C.mean_pairwise_distance(np.stack([a])) is None


def test_robust_z_mad_zero_falls_back_to_mean_deviation():
    d = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9])
    z = C.robust_z(d)
    assert z[-1] > 3.5 and np.allclose(z[:-1], 0)
    assert np.allclose(C.robust_z(np.full(5, 0.2)), 0)


def _kad_bruteforce(X, Y, sigma, alpha=100.0):
    k = lambda x, y: np.exp(-np.sum((x - y) ** 2) / (2 * sigma**2))
    m, n = len(X), len(Y)
    xx = sum(k(X[i], X[j]) for i in range(m) for j in range(m) if i != j) / (m * (m - 1))
    yy = sum(k(Y[i], Y[j]) for i in range(n) for j in range(n) if i != j) / (n * (n - 1))
    xy = sum(k(x, y) for x in X for y in Y) / (m * n)
    return alpha * (xx + yy - 2 * xy)


def test_kad_matches_the_unbiased_estimator_and_median_bandwidth(rng):
    X = _cluster(_unit(rng.standard_normal(DIM)), 6, 0.4, rng)
    Y = _cluster(_unit(rng.standard_normal(DIM)), 7, 0.4, rng)
    out = C.kad(X, Y)
    iu = np.triu_indices(len(Y), k=1)
    sigma = np.median(np.linalg.norm(Y[:, None] - Y[None], axis=-1)[iu])
    assert out["bandwidth"] == pytest.approx(sigma)
    assert out["kad"] == pytest.approx(_kad_bruteforce(X, Y, sigma), rel=1e-9)


def test_kad_near_zero_for_one_distribution_and_large_for_two(rng):
    a = _unit(rng.standard_normal(DIM))
    b = _unit(rng.standard_normal(DIM))
    same = C.kad(_cluster(a, 30, 0.3, rng), _cluster(a, 30, 0.3, rng))["kad"]
    diff = C.kad(_cluster(b, 30, 0.3, rng), _cluster(a, 30, 0.3, rng))["kad"]
    assert abs(same) < 2.0
    assert diff > 20 * max(abs(same), 0.1)
    with pytest.raises(ValueError):
        C.kad(np.stack([a]), _cluster(a, 3, 0.3, rng))


# --- coverage -------------------------------------------------------------------------------

def _brief():
    return CV.load_brief({"families": {
        "kick": {"quota": 2, "text": ["kick a", "kick b"], "tag": "_k_"},
        "hat": {"quota": 1, "max": 1, "text": "hat", "tag": "hat"},
        "shaker": {"quota": 1, "text": "shaker", "tag": "shaker"},
    }})


def test_coverage_text_mode_reports_the_missing_family_and_the_overflow(rng):
    k, h, s = (_unit(rng.standard_normal(DIM)) for _ in range(3))
    text = {"kick a": _unit(k + 0.1 * rng.standard_normal(DIM)),
            "kick b": _unit(k + 0.1 * rng.standard_normal(DIM)), "hat": h, "shaker": s}
    X = np.vstack([_cluster(k, 3, 0.3, rng), _cluster(h, 2, 0.3, rng)])  # no shaker; 2 hats
    brief = _brief()
    names = list(brief["families"])
    poles = np.stack([V.pole([text[p] for p in brief["families"][n]["text"]]) for n in names])
    idx, _top, margin = CV.assign_text(X, poles)
    assigned = [names[i] for i in idx]
    assert assigned == ["kick"] * 3 + ["hat"] * 2
    assert (margin > 0).all()
    t = CV.tally(names, assigned, brief, [f"s{i}" for i in range(5)])
    assert t["missing"] == ["shaker"]
    assert t["over"] == ["hat"]
    assert t["families"]["kick"]["delta"] == 1 and t["families"]["shaker"]["delta"] == -1


def test_coverage_frames_tag_mode_and_unassigned(rng):
    labels = ["01_k_sine", "02_k_knock", "03_m_hat", "04_s_rim"]
    items = [V.Item(id=f"a{i}", vector_id=f"v{i}", label=lab, model="m",
                    vec=_unit(rng.standard_normal(DIM))) for i, lab in enumerate(labels)]
    frames = CV.coverage_frames(items, _brief(), model="m", mode="tag")
    pack = frames[-1]["data"]
    assert pack["coverage.families"]["kick"]["count"] == 2
    assert pack["coverage.missing"] == ["shaker"]
    assert pack["coverage.unassigned"] == ["04_s_rim"]
    assert [f["data"]["coverage.family"] for f in frames[:-1]] == ["kick", "kick", "hat", None]
    assert all(f["of"] == it.id for f, it in zip(frames, items))


def test_default_mode_needs_text_or_tags_everywhere():
    assert CV.default_mode(_brief()) == "text"
    tags_only = CV.load_brief({"families": {"a": {"tag": "a"}, "b": {"tag": "b"}}})
    assert CV.default_mode(tags_only) == "tag"
    with pytest.raises(ValueError):
        CV.default_mode(CV.load_brief({"families": {"a": {"tag": "a"}, "b": {"text": "b"}}}))


# --- probe ----------------------------------------------------------------------------------

def test_probe_contrast_orders_dirty_over_clean_where_raw_cosine_does_not(rng):
    """The calibration's point: a shared 'sound-ness' direction dominates raw cosines."""
    c = _unit(rng.standard_normal(DIM))           # what every prompt and sample shares
    d = _unit(rng.standard_normal(DIM))
    d = _unit(d - (d @ c) * c)                    # the dirt direction, orthogonal to c
    e = _unit(rng.standard_normal(DIM))
    e = _unit(e - (e @ c) * c - (e @ d) * d)      # unrelated detail
    clean = _unit(1.0 * c)
    dirty = _unit(0.6 * c + 0.3 * d + 0.5 * e)
    pos, neg = _unit(c + 0.5 * d), _unit(c - 0.5 * d)
    assert clean @ pos > dirty @ pos              # raw cosine ranks the clean one "dirtier"
    ax = P.axis([pos], [neg])
    s, z = P.scores(np.stack([clean, dirty]), ax)
    assert s[1] > s[0]                            # the contrast gets it right
    assert z is None                              # no in-set z below 3 items


def test_probe_frames_rank_and_z(rng):
    c = _unit(rng.standard_normal(DIM))
    d = _unit(rng.standard_normal(DIM))
    X = np.stack([_unit(c + w * d) for w in (0.0, 0.2, 0.5, 1.0)])
    items = [V.Item(id=f"a{i}", vector_id=f"v{i}", label=f"w{i}", model="m", vec=x)
             for i, x in enumerate(X)]
    frames = P.probe_frames(items, pos=["dirty"], neg=["clean"],
                            text_vecs={"dirty": _unit(c + d), "clean": _unit(c - d)}, model="m")
    ranked = frames[-1]["data"]["probe.ranked"]
    assert [r["label"] for r in ranked] == ["w3", "w2", "w1", "w0"]
    zs = [f["data"]["probe.z"] for f in frames[:-1]]
    assert zs == sorted(zs) and abs(sum(zs)) < 5e-3  # rounded to 3 places


def test_pole_pools_normalized_prompts():
    a, b = np.array([3.0, 0, 0]), np.array([0, 0.1, 0])
    assert np.allclose(V.pole([a, b]), _unit([1, 1, 0]))  # each prompt counts once, not by norm
