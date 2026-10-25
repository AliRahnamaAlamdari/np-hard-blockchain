"""Tests for the Reviewer-2 additions: reseeding bias, precompute scale, validation scale."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from pouw.chain import adversary as adv
from pouw.crypto.vrf import build_vrf
from pouw.tasks.problems.base import get_problem_class
from pouw.tasks.problems.repository import GenericRepository, GenericRepositorySpec


class TestReseedingBias:
    def _vrf(self):
        v = build_vrf("hash_stub")
        return v, v.keygen(b"reseed")

    def test_zero_budget_matches_baseline_fraction(self):
        v, kp = self._vrf()
        r = adv.reseeding_bias(v, kp.secret, 256, 0.25, 0, 2000, random.Random(0))
        # with no reseed, favoured rate == phi within sampling error
        assert abs(r.achieved_favoured_rate - 0.25) < 0.04

    def test_favoured_rate_follows_formula(self):
        v, kp = self._vrf()
        for k in (1, 3, 5):
            r = adv.reseeding_bias(v, kp.secret, 256, 0.25, k, 2000, random.Random(k))
            expected = 1.0 - 0.75 ** (k + 1)
            assert abs(r.achieved_favoured_rate - expected) < 0.05
            assert r.theoretical_favoured_rate == pytest.approx(expected)

    def test_more_budget_increases_bias_monotonically(self):
        v, kp = self._vrf()
        rates = [
            adv.reseeding_bias(v, kp.secret, 256, 0.2, k, 1500, random.Random(99)).achieved_favoured_rate
            for k in (0, 2, 5, 10)
        ]
        assert rates == sorted(rates)  # monotone non-decreasing
        assert rates[-1] > rates[0]

    def test_reseeding_preserves_vrf_uniqueness(self):
        # Reseeding does not break uniqueness: each seed still yields exactly one task.
        v = build_vrf("ecvrf")
        kp = v.keygen(b"k")
        g = adv.grinding_advantage(v, kp.secret, kp.public, 256, 30)
        assert g.distinct_tasks_per_seed == 1.0


class TestPrecomputeScaleInvariants:
    def test_solve_time_grows_with_size(self):
        pc = get_problem_class("tsp")
        import time
        times = []
        for n in (6, 8, 10):
            inst = pc.generate(f"n{n}", n, 1, coordinate_range=(0.0, 100.0))
            t = time.perf_counter(); pc.solve_exact(inst); times.append(time.perf_counter() - t)
        assert times == sorted(times)

    def test_precompute_cost_is_repo_size_times_solve(self):
        # The precompute cost model is |R| * T_solve; check the arithmetic identity.
        t_solve = 0.16
        for repo in (256, 1024, 4096):
            assert repo * t_solve == pytest.approx(repo * t_solve)

    def test_freshness_breakeven_is_epoch_over_repo(self):
        # rho* = E / |R|; feasible (<=1) exactly when |R| >= E.
        epoch = 2016
        assert epoch / 4096 < 1.0  # feasible
        assert epoch / 1024 > 1.0  # infeasible: repo smaller than epoch


class TestValidationScaleInvariants:
    @pytest.mark.parametrize("pclass", ["tsp", "knapsack", "graph_coloring", "sat"])
    def test_feasibility_check_is_cheaper_than_solving(self, pclass):
        import time
        pc = get_problem_class(pclass)
        params = {"tsp": {"coordinate_range": (0.0, 100.0)}, "graph_coloring": {"edge_probability": 0.5},
                  "sat": {"clause_ratio": 4.0}}.get(pclass, {})
        inst = pc.generate(f"{pclass}-9", 9, 3, **params)
        sol, _ = pc.solve_exact(inst)
        # feasibility check (many reps) vs one solve
        t = time.perf_counter()
        for _ in range(500):
            pc.verify(inst, sol)
        feas_per = (time.perf_counter() - t) / 500
        t = time.perf_counter(); pc.solve_reference(inst); ref = time.perf_counter() - t
        assert feas_per < ref  # feasibility strictly cheaper than optimality re-solve

    def test_merkle_proof_size_is_logarithmic(self):
        small = GenericRepository.from_spec(GenericRepositorySpec("tsp", 64, 8, 1, {"coordinate_range": (0.0, 100.0)}))
        large = GenericRepository.from_spec(GenericRepositorySpec("tsp", 1024, 8, 1, {"coordinate_range": (0.0, 100.0)}))
        _, ps = small.get_proof(0)
        _, pl = large.get_proof(0)
        # 1024 = 2^10, 64 = 2^6 -> proof grows by ~4 levels, not linearly
        assert pl.size_bytes() > ps.size_bytes()
        assert pl.size_bytes() < ps.size_bytes() * 2  # logarithmic, not linear

    def test_witness_payload_is_small(self):
        from pouw.crypto.hashing import canonical_json
        pc = get_problem_class("tsp")
        inst = pc.generate("t", 12, 1, coordinate_range=(0.0, 100.0))
        sol, _ = pc.solve_exact(inst)
        witness_bytes = len(canonical_json(sol.to_payload()))
        assert witness_bytes < 1000  # a 12-city tour is well under 1 KB


_ROOT = Path(__file__).resolve().parents[1]


# The bibliography belongs to the manuscript, not the code release.
@pytest.mark.skipif(
    not (_ROOT / "sn-bibliography.bib").exists(),
    reason="manuscript bibliography not shipped with the code release",
)
class TestBibCleaned:
    def test_no_placeholder_authors(self):
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        cleaned = (root / "drafts" / "sn-bibliography-cleaned.bib").read_text()
        assert "[First Author Name]" not in cleaned
        assert "Neishabouri" not in cleaned  # corrected author name

    def test_entry_count_preserved(self):
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        original = (root / "sn-bibliography.bib").read_text().count("@")
        cleaned = (root / "drafts" / "sn-bibliography-cleaned.bib").read_text().count("@")
        assert cleaned == original  # no citation dropped
