
from blockchain.chain import Blockchain
from vrf.vrf import SimpleVRF
from registry.registry import Registry
from solver.tsp_solver import TSPSolver
from proof.proof import SimpleProof

class Node:
    def __init__(self, node_id: int, difficulty: int, repo: Registry):
        self.node_id = node_id
        self.blockchain = Blockchain(difficulty=difficulty)
        # برای سادگی، کلید ماینر فقط یک رشته است
        self.miner_pk = f"miner-{node_id}"
        # رجیستری مشترک
        self.repo = repo

    def validate_and_add_block(self, block, vrf_engine: SimpleVRF, proof_module: SimpleProof):
        ok, msg = self.blockchain.validate_block(block, vrf_engine=vrf_engine, repo=self.repo, proof_module=proof_module)
        if ok:
            self.blockchain.chain.append(block)
        return ok, msg
