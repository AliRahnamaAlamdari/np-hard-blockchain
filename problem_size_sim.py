import os
import time
from registry.registry import Registry
from blockchain.block import Block
from vrf.vrf import SimpleVRF
from solver.tsp_solver import TSPSolver
from proof.proof import SimpleProof
from network.node import Node

class ProblemSizeSimulator:
    def __init__(self, difficulty: int, repo: Registry):
        self.repo = repo
        self.difficulty = difficulty
        self.vrf_engine = SimpleVRF()
        self.solver = TSPSolver()
        self.proof = SimpleProof(mode="reveal")
        # 4 Nodes are enough for this benchmark
        self.nodes = [Node(node_id=i, difficulty=difficulty, repo=repo) for i in range(4)]

    def run_benchmark_for_instance(self, instance: dict, num_blocks: int = 3):
        total_pow_time = 0.0
        total_work_time = 0.0

        # Reset chains for clean test
        for node in self.nodes:
            node.blockchain.chain = [node.blockchain.chain[0]]

        for h in range(num_blocks):
            tip = self.nodes[0].blockchain.get_last_block()
            index = tip.index + 1
            prev_hash = tip.hash_value
            tx_root = f"txroot-{h}"

            # --- 1. Measure PoW Time ---
            start_pow = time.perf_counter()
            winner_node = None
            winner_nonce = None
            winner_pow_hash = None
            nonce = 0
            
            while True:
                for node in self.nodes:
                    pow_hash = node.blockchain.calculate_pow_hash(index, prev_hash, tx_root, nonce, node.miner_pk)
                    if pow_hash.startswith("0" * self.difficulty):
                        winner_node = node
                        winner_nonce = nonce
                        winner_pow_hash = pow_hash
                        break
                if winner_node is not None:
                    break
                nonce += 1
            total_pow_time += (time.perf_counter() - start_pow)

            # --- 2. Measure Useful Work Time (Forced to specific instance) ---
            start_work = time.perf_counter()
            solution, solution_cost = self.solver.solve(instance)
            commitment = self.proof.commit_solution(solution) if self.proof else None
            total_work_time += (time.perf_counter() - start_work)

            # Construct and broadcast block to advance the chain
            vrf_y, vrf_proof = self.vrf_engine.prove(winner_pow_hash)
            block = Block(
                index, prev_hash, tx_root, winner_nonce, winner_node.miner_pk, winner_pow_hash,
                vrf_y, vrf_proof, self.repo.data.get("repo_id"), self.repo.data.get("version"),
                instance["instance_id"], None, solution, solution_cost, commitment
            )
            for node in self.nodes:
                node.blockchain.chain.append(block)

        return total_pow_time / num_blocks, total_work_time / num_blocks