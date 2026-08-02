"""End-to-end pipeline tests plus the sanity checks a reviewer would want.

Every experiment is executed on the smoke config so the whole path -- config,
seeding, simulation, metrics, serialisation, LaTeX, figures -- is exercised.  Marked
``slow`` so the fast suite stays quick::

    pytest -m "not slow"   # unit tests only
    pytest                 # everything
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pouw.config import load_config
from pouw.experiments import PAPER_MAP, REGISTRY
from pouw.reporting.io import environment_fingerprint, write_csv, write_json

ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def smoke_config_path() -> str:
    return str(ROOT / "configs" / "smoke.yaml")


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_experiment_runs_end_to_end(name, smoke_config_path, tmp_path):
    """Each experiment produces a JSON payload with a manifest and no empty tables."""
    config = load_config(
        smoke_config_path, overrides={"run": {"output_dir": str(tmp_path)}}
    )
    output = REGISTRY[name](config).execute()

    assert output.name == name
    assert output.files, "experiment wrote no files"

    json_files = [path for path in output.files if path.suffix == ".json"]
    assert json_files, "experiment wrote no JSON result"
    document = json.loads(json_files[0].read_text())

    assert "manifest" in document and "results" in document
    assert document["manifest"]["config"]["name"] == "smoke"
    assert document["manifest"]["environment"]["python_version"]
    assert document["results"]["meta"]["experiment"] == name
    assert document["results"]["meta"]["wall_seconds"] > 0

    tables = document["results"].get("csv_tables", {})
    for table_name, rows in tables.items():
        assert rows, f"{name}: csv table {table_name} is empty"


def test_every_experiment_is_mapped_to_a_paper_artifact():
    assert set(PAPER_MAP) == set(REGISTRY)


class TestSanityChecks:
    """Domain sanity checks: the analogue of 'loss decreases' for this pipeline."""

    @staticmethod
    @pytest.fixture(scope="class")
    def results(tmp_path_factory):
        """Run the experiments needed by these checks once, and share the payloads."""
        base = tmp_path_factory.mktemp("sanity")
        payloads = {}
        for name in (
            "table3_time_allocation",
            "table4_problem_size",
            "ablation_vrf_grinding",
            "convergence",
            "sensitivity_solver_advantage",
        ):
            config = load_config(
                str(ROOT / "configs" / "smoke.yaml"),
                overrides={"run": {"output_dir": str(base)}},
            )
            payloads[name] = REGISTRY[name](config).execute().payload
        return payloads

    def test_useful_share_decreases_with_difficulty(self, results):
        """The framework's central tuning claim (G1) must hold monotonically."""
        rows = results["table3_time_allocation"]["table_rows"]
        shares = [row["useful_share_percent"] for row in rows]
        assert shares == sorted(shares, reverse=True), (
            f"useful share should fall as PoW difficulty rises, got {shares}"
        )

    def test_pow_time_increases_with_difficulty(self, results):
        rows = results["table3_time_allocation"]["table_rows"]
        pow_times = [row["mean_pow_seconds"] for row in rows]
        assert pow_times == sorted(pow_times)

    def test_shares_are_bounded_and_complementary(self, results):
        for row in results["table3_time_allocation"]["table_rows"]:
            assert 0.0 <= row["useful_share_percent"] <= 100.0
            assert row["pow_share_percent"] + row["useful_share_percent"] == pytest.approx(100.0)

    def test_solve_time_grows_with_instance_size(self, results):
        rows = results["table4_problem_size"]["table_rows"]
        times = [row["mean_solve_seconds"] for row in rows]
        assert times == sorted(times), f"solve time should grow with N, got {times}"

    def test_tour_counts_match_factorial(self, results):
        import math

        for row in results["table4_problem_size"]["table_rows"]:
            assert row["tours_enumerated"] == math.factorial(row["size"] - 1)
            assert row["mean_solver_operations"] == pytest.approx(row["tours_enumerated"])

    def test_held_karp_finds_the_same_optimum(self, results):
        for row in results["table4_problem_size"]["table_rows"]:
            if "heldkarp_matches_optimum" in row:
                assert row["heldkarp_matches_optimum"]

    def test_real_vrf_admits_exactly_one_task_per_seed(self, results):
        rows = {row["vrf_backend"]: row for row in results["ablation_vrf_grinding"]["table_rows"]}
        assert rows["ecvrf"]["distinct_tasks_max_eval"] == 1
        assert rows["ecvrf"]["unique_output"]

    def test_malleable_vrf_admits_many_tasks_per_seed(self, results):
        rows = {row["vrf_backend"]: row for row in results["ablation_vrf_grinding"]["table_rows"]}
        assert rows["ecdsa_pseudo"]["distinct_tasks_max_eval"] > 1
        # The attack only counts if the forged proofs actually validate.
        assert rows["ecdsa_pseudo"]["all_proofs_verified"]

    def test_all_nodes_converge_and_solutions_are_optimal(self, results):
        summary = results["convergence"]["summary"]
        assert summary["all_converged"]
        assert summary["acceptance_rate_mean"] == pytest.approx(1.0)
        if "optimality_rate_mean" in summary:
            assert summary["optimality_rate_mean"] == pytest.approx(1.0)

    def test_better_solver_is_faster_but_equally_correct(self, results):
        for row in results["sensitivity_solver_advantage"]["table_rows"]:
            assert row["same_optimum"], "both solvers are exact; they must agree"
            if row["size"] >= 8:
                assert row["speedup"] > 1.0


class TestSerialisation:
    def test_json_records_a_manifest(self, tmp_path):
        path = write_json(tmp_path / "r.json", {"value": 1}, config=None)
        document = json.loads(path.read_text())
        assert "generated_at" in document["manifest"]
        assert document["results"]["value"] == 1

    def test_non_finite_values_survive_serialisation(self, tmp_path):
        path = write_json(tmp_path / "r.json", {"ratio": float("inf")})
        assert json.loads(path.read_text())["results"]["ratio"] == "inf"

    def test_csv_writes_a_header_and_rows(self, tmp_path):
        path = write_csv(tmp_path / "r.csv", [{"a": 1, "b": 2}, {"a": 3, "b": 4}])
        lines = path.read_text().strip().splitlines()
        assert lines[0] == "a,b"
        assert len(lines) == 3

    def test_empty_csv_is_written_without_error(self, tmp_path):
        assert write_csv(tmp_path / "e.csv", []).read_text() == ""

    def test_environment_fingerprint_reports_versions(self):
        fingerprint = environment_fingerprint()
        assert fingerprint["python_version"]
        assert "numpy_version" in fingerprint


class TestLatexOutput:
    def test_tables_are_wellformed_environments(self):
        from pouw.reporting.latex import generic_table

        content = generic_table(
            headers=["A", "B"], rows=[["1", "2"]], caption="Cap", label="tab:x"
        )
        assert content.startswith("\\begin{table}")
        assert content.rstrip().endswith("\\end{table}")
        assert content.count("\\toprule") == 1
        assert content.count("\\bottomrule") == 1
        assert "\\label{tab:x}" in content

    def test_non_finite_cells_render_as_symbols(self):
        from pouw.reporting.latex import number

        assert number(float("inf")) == "$\\infty$"
        assert number("inf") == "$\\infty$"
        assert number(float("nan")) == "---"
        assert number(None) == "---"
        assert number(1.5) == "1.5000"

    def test_specials_are_escaped_when_escaping_is_enabled(self):
        from pouw.reporting.latex import tabular

        content = tabular(
            headers=["A"],
            rows=[["a_b & 50%"]],
            column_spec="l",
            caption="C",
            label="tab:y",
            escape_cells=True,
        )
        body = content.split("\\midrule")[1]
        assert "a\\_b" in body and "\\&" in body and "\\%" in body

    def test_escaping_is_off_for_prebuilt_math_cells(self):
        """Result builders pass LaTeX math through, so generic_table must not escape."""
        from pouw.reporting.latex import generic_table

        content = generic_table(
            headers=["A"], rows=[["$12.5\\times$"]], caption="C", label="tab:z"
        )
        assert "$12.5\\times$" in content
