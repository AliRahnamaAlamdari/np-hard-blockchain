from blockchain.chain import Blockchain
from vrf.vrf import SimpleVRF
from registry.registry import Registry
from solver.tsp_solver import TSPSolver
from proof.proof import SimpleProof
import os

def run_simulation():
    repo_path = os.path.join(os.path.dirname(__file__), "registry", "registry.json")
    repo = Registry(repo_path)
    solver = TSPSolver()
    proof = SimpleProof(mode="reveal")
    vrf = SimpleVRF()
    bc = Blockchain(difficulty=3)

    miner_pk = vrf.get_public_key_hex()

    for i in range(3):
        print("\n=== MINING ROUND", i, "===")
        block = bc.mine_block(miner_pk, tx_root=f"txroot-{i}", repo_id=repo.data.get("repo_id"),
                              repo_version=repo.data.get("version"), vrf_engine=vrf, repo=repo, solver=solver, proof_module=proof)
        print("Mined block index:", block.index)
        print("instance:", block.instance_id, "solution_cost:", block.solution_cost)
        ok, msg = bc.validate_block(block, vrf_engine=vrf, repo=repo, proof_module=proof)
        print("Validation result:", ok, msg)

if __name__ == "__main__":
    run_simulation()
