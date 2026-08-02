"""Determinism and configuration tests.

Reproducibility is the point of this repository, so seeding is tested as a first-class
property: the same seed must give bit-identical results, and different seeds must
actually differ (a seed that is silently ignored would pass the first test alone).
"""

from __future__ import annotations

import pytest

from pouw.chain.network import Network
from pouw.config import ExperimentConfig, load_config, validate
from pouw.checkpoint import Checkpoint, clear, config_fingerprint, load, save
from pouw.seeding import derive_seed, seed_everything
from pouw.tasks.repository import ProblemRepository, RepositorySpec


class TestSeedDerivation:
    def test_same_inputs_give_the_same_sub_seed(self):
        assert derive_seed(7, "miner:0") == derive_seed(7, "miner:0")

    def test_different_labels_give_different_sub_seeds(self):
        assert derive_seed(7, "miner:0") != derive_seed(7, "miner:1")

    def test_different_run_seeds_give_different_sub_seeds(self):
        assert derive_seed(7, "miner:0") != derive_seed(8, "miner:0")

    def test_sub_seeds_fit_in_64_bits(self):
        assert 0 <= derive_seed(123, "x") < 2**64

    def test_bundle_streams_are_independent_of_call_order(self):
        first = seed_everything(5)
        a1 = first.python_random("alpha").random()
        b1 = first.python_random("beta").random()
        second = seed_everything(5)
        b2 = second.python_random("beta").random()
        a2 = second.python_random("alpha").random()
        assert (a1, b1) == (a2, b2)

    def test_key_material_is_reproducible(self):
        assert seed_everything(3).bytes_for("miner:0") == seed_everything(3).bytes_for("miner:0")

    def test_numpy_generator_is_reproducible(self):
        first = seed_everything(9).numpy_generator("noise").random(5)
        second = seed_everything(9).numpy_generator("noise").random(5)
        assert (first == second).all()


class TestRepositoryDeterminism:
    def test_same_spec_gives_an_identical_repository(self):
        spec = RepositorySpec(
            num_instances=8, instance_size=5, seed=42, coordinate_range=(0.0, 100.0)
        )
        first = ProblemRepository.from_spec(spec)
        second = ProblemRepository.from_spec(spec)
        assert first.root == second.root
        assert [i.to_payload() for i in first] == [i.to_payload() for i in second]


class TestSimulationDeterminism:
    def _run(self, seed: int, repository: ProblemRepository):
        network = Network(
            num_miners=3,
            repository=repository,
            difficulty_bits=8,
            vrf_backend="ecvrf",
            seeds=seed_everything(seed),
            solver="brute_force",
        )
        return network.run(num_blocks=5, log_every=0)

    def test_same_seed_reproduces_the_same_chain(self, small_repository):
        first = self._run(11, small_repository)
        second = self._run(11, small_repository)
        assert [r.block_hash for r in first.blocks] == [r.block_hash for r in second.blocks]
        assert first.blocks_won == second.blocks_won

    def test_same_seed_reproduces_the_same_task_assignments(self, small_repository):
        first = self._run(11, small_repository)
        second = self._run(11, small_repository)
        assert [r.instance_id for r in first.blocks] == [r.instance_id for r in second.blocks]

    def test_different_seeds_produce_different_chains(self, small_repository):
        first = self._run(11, small_repository)
        second = self._run(22, small_repository)
        assert [r.block_hash for r in first.blocks] != [r.block_hash for r in second.blocks]

    def test_solution_costs_are_reproducible(self, small_repository):
        first = self._run(11, small_repository)
        second = self._run(11, small_repository)
        assert [r.solution_cost for r in first.blocks] == [r.solution_cost for r in second.blocks]


class TestConfigLoading:
    def test_default_config_loads_and_validates(self):
        config = load_config("configs/default.yaml")
        assert config.name == "default"
        assert config.pow.difficulty_bits > 0

    def test_smoke_config_inherits_from_default(self):
        config = load_config("configs/smoke.yaml")
        # Overridden in smoke.yaml
        assert config.network.num_miners == 3
        # Inherited from default.yaml
        assert config.solver.name == "brute_force"

    @pytest.mark.parametrize(
        "name",
        [
            "table3_time_allocation",
            "table4_problem_size",
            "table5_fairness",
            "table6_validation_cost",
            "convergence",
            "ablation_vrf_grinding",
            "ablation_components",
            "sensitivity_repository_size",
            "sensitivity_hash_rate",
            "sensitivity_solver_advantage",
        ],
    )
    def test_every_experiment_config_loads(self, name):
        config = load_config(f"configs/experiments/{name}.yaml")
        assert config.name == name
        validate(config)

    def test_overrides_take_precedence(self):
        config = load_config("configs/default.yaml", overrides={"run": {"seeds": [99]}})
        assert config.run.seeds == [99]

    def test_unknown_key_is_rejected(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("pow:\n  not_a_real_key: 1\n")
        with pytest.raises(ValueError, match="unknown config key"):
            load_config(path)

    def test_missing_config_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("configs/does_not_exist.yaml")


class TestConfigValidation:
    def test_bad_difficulty_is_rejected(self):
        config = ExperimentConfig()
        config.pow.difficulty_bits = 999
        with pytest.raises(ValueError, match="difficulty_bits"):
            validate(config)

    def test_bad_vrf_backend_is_rejected(self):
        config = ExperimentConfig()
        config.vrf.backend = "nope"
        with pytest.raises(ValueError, match="vrf.backend"):
            validate(config)

    def test_bad_solver_is_rejected(self):
        config = ExperimentConfig()
        config.solver.name = "nope"
        with pytest.raises(ValueError, match="solver.name"):
            validate(config)

    def test_hash_rate_length_mismatch_is_rejected(self):
        config = ExperimentConfig()
        config.network.num_miners = 3
        config.network.hash_rates = [1.0, 2.0]
        with pytest.raises(ValueError, match="hash_rates"):
            validate(config)

    def test_empty_seed_list_is_rejected(self):
        config = ExperimentConfig()
        config.run.seeds = []
        with pytest.raises(ValueError, match="seeds"):
            validate(config)

    def test_instance_size_below_three_is_rejected(self):
        config = ExperimentConfig()
        config.repository.instance_size = 2
        with pytest.raises(ValueError, match="instance_size"):
            validate(config)


class TestCheckpointing:
    def _checkpoint(self, config_hash: str) -> Checkpoint:
        return Checkpoint(
            experiment="demo",
            config_hash=config_hash,
            seed=0,
            completed_blocks=10,
            records=[{"height": 1}],
            state={},
        )

    def test_save_and_load_roundtrip(self, tmp_path):
        path = tmp_path / "cp.json"
        fingerprint = "abc123"
        save(path, self._checkpoint(fingerprint))
        loaded = load(path, fingerprint)
        assert loaded is not None
        assert loaded.completed_blocks == 10

    def test_checkpoint_from_a_different_config_is_ignored(self, tmp_path):
        path = tmp_path / "cp.json"
        save(path, self._checkpoint("abc123"))
        assert load(path, "different-hash") is None

    def test_missing_checkpoint_returns_none(self, tmp_path):
        assert load(tmp_path / "absent.json", "x") is None

    def test_corrupt_checkpoint_returns_none(self, tmp_path):
        path = tmp_path / "cp.json"
        path.write_text("{not json")
        assert load(path, "x") is None

    def test_clear_removes_the_file(self, tmp_path):
        path = tmp_path / "cp.json"
        save(path, self._checkpoint("abc"))
        clear(path)
        assert not path.exists()

    def test_fingerprint_is_stable_and_sensitive(self):
        config = ExperimentConfig()
        first = config_fingerprint(config)
        assert first == config_fingerprint(ExperimentConfig())
        config.pow.difficulty_bits += 1
        assert config_fingerprint(config) != first
