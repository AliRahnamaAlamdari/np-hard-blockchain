
import random
from typing import List, Dict, Any, Tuple
from utils.logs import BlockEventLog
import time


from blockchain.block import Block
from blockchain.chain import Blockchain
from vrf.vrf import SimpleVRF
from registry.registry import Registry
from solver.tsp_solver import TSPSolver
from proof.proof import SimpleProof

from .node import Node

class NetworkSimulator:
    def __init__(self, num_nodes: int, difficulty: int, repo: Registry):
        self.repo = repo
        self.difficulty = difficulty
        self.vrf_engine = SimpleVRF()
        self.solver = TSPSolver()
        self.proof = SimpleProof(mode="reveal")
        self.nodes: List[Node] = [
            Node(node_id=i, difficulty=difficulty, repo=repo) for i in range(num_nodes)
        ]

    # def _make_random_txs(self, count: int = 5) -> List[Dict[str, Any]]:
    #     txs = []
    #     for _ in range(count):
    #         src = f"user{random.randint(1,5)}"
    #         dst = f"user{random.randint(6,10)}"
    #         amount = random.randint(1, 50)
    #         txs.append({"from": src, "to": dst, "amount": amount})
    #     return txs

    def _construct_block(self, index: int, prev_hash: str, tx_root: str,
                         miner_pk: str, nonce: int, pow_hash: str) -> Block:
        vrf_y = vrf_proof = None
        instance_id = merkle_proof = None
        solution = solution_cost = commitment = None

        if self.vrf_engine and self.repo:
            vrf_y, vrf_proof = self.vrf_engine.prove(pow_hash)
            vrf_int = int(vrf_y, 16)
            n = self.repo.size()
            idx = vrf_int % n
            instance_id = self.repo.get_instance_id(idx)
            instance, merkle_proof = self.repo.get_instance_with_proof(idx)
            if self.solver:
                solution, solution_cost = self.solver.solve(instance)
                if self.proof:
                    commitment = self.proof.commit_solution(solution)



        repo_id = self.repo.data.get("repo_id")
        repo_version = self.repo.data.get("version")

        block = Block(
            index,
            prev_hash,
            tx_root,
            nonce,
            miner_pk,
            pow_hash,
            vrf_y,
            vrf_proof,
            repo_id,
            repo_version,
            instance_id,
            merkle_proof,
            solution,
            solution_cost,
            commitment,
        )
        return block

    def mine_one_block(self, height: int):
        log = BlockEventLog(height)
     
        # فرض: همه نودها زنجیره هم‌ارتفاع دارند
        tip = self.nodes[0].blockchain.get_last_block()
        log.add("Tip Hash", tip.hash_value[:16])
        log.add("Difficulty", self.difficulty)
        log.add("Total Miners", len(self.nodes))
        index = tip.index + 1
        prev_hash = tip.hash_value

        # txs = self._make_random_txs(count=6)
        tx_root = f"txroot-{height}"

        winner_node = None
        winner_nonce = None
        winner_pow_hash = None

        nonce = 0
        print(f"\n=== Mining height {index} ===")
        while True:
            for node in self.nodes:
                pow_hash = node.blockchain.calculate_pow_hash(index, prev_hash, tx_root, nonce, node.miner_pk)
                if pow_hash.startswith("0" * self.difficulty):
                    winner_node = node
                    log.add("PoW Winner", winner_node.node_id)
                    winner_nonce = nonce
                    winner_pow_hash = pow_hash
                    print(f"🏆 Winner node {node.node_id} at nonce={nonce}, pow={pow_hash[:16]}...")
                    break
            if winner_node is not None:
                break
            nonce += 1

        block = self._construct_block(index, prev_hash, tx_root, winner_node.miner_pk, winner_nonce, winner_pow_hash)

        # broadcast
        for node in self.nodes:
            ok, msg = node.validate_and_add_block(block, vrf_engine=self.vrf_engine, proof_module=self.proof)
            if not ok:
                print(f"❌ Node {node.node_id} rejected block {index}: {msg}")
        return block

    def run(self, num_blocks: int = 3):
        for h in range(num_blocks):
            self.mine_one_block(h)

    def print_chains(self):
        print("\n📊 Final chains:")
        last_hashes = set()
        for node in self.nodes:
            print(f"\n🧩 Node {node.node_id} chain:")
            for blk in node.blockchain.chain:
                print(
                    f"  ▪ Height {blk.index} | Hash {blk.hash_value[:12]}... | Miner {blk.miner_pk} | cost={blk.solution_cost}"
                )
            last_hashes.add(node.blockchain.get_last_block().hash_value)

        print("\n------------------------------")
        if len(last_hashes) == 1:
            print("✅ All nodes synchronized — same tip hash.")
        else:
            print("⚠️ Chains diverged!")
