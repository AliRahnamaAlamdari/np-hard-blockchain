import os
import math
import time
from typing import List
from registry.registry import Registry
from blockchain.block import Block
from vrf.vrf import SimpleVRF
from solver.tsp_solver import TSPSolver
from proof.proof import SimpleProof
from network.node import Node

class FairnessNetworkSimulator:
    def __init__(self, num_nodes: int, difficulty: int, repo: Registry):
        self.repo = repo
        self.difficulty = difficulty
        self.vrf_engine = SimpleVRF()
        self.solver = TSPSolver()
        self.proof = SimpleProof(mode="reveal")
        self.nodes: List[Node] = [
            Node(node_id=i, difficulty=difficulty, repo=repo) for i in range(num_nodes)
        ]
        
        # دیکشنری برای شمارش تعداد بلاک‌های برنده شده توسط هر ماینر
        self.block_counts = {node.node_id: 0 for node in self.nodes}

    def _construct_block(self, index: int, prev_hash: str, tx_root: str,
                         miner_pk: str, nonce: int, pow_hash: str) -> Block:
        # برای سرعت بالا در این آزمایش، کار مفید (TSP) را به صورت ساده فراخوانی می‌کنیم
        vrf_y = vrf_proof = None
        instance_id = merkle_proof = None
        solution = solution_cost = commitment = None

        if self.vrf_engine and self.repo:
            vrf_y, vrf_proof = self.vrf_engine.prove(pow_hash)
            vrf_int = int(vrf_y, 16)
            idx = vrf_int % self.repo.size()
            instance_id = self.repo.get_instance_id(idx)
            instance, merkle_proof = self.repo.get_instance_with_proof(idx)
            if self.solver:
                solution, solution_cost = self.solver.solve(instance)
                if self.proof:
                    commitment = self.proof.commit_solution(solution)

        return Block(
            index, prev_hash, tx_root, nonce, miner_pk, pow_hash,
            vrf_y, vrf_proof, self.repo.data.get("repo_id"), self.repo.data.get("version"),
            instance_id, merkle_proof, solution, solution_cost, commitment,
        )

    def mine_one_block(self, height: int):
        tip = self.nodes[0].blockchain.get_last_block()
        index = tip.index + 1
        prev_hash = tip.hash_value
        tx_root = f"txroot-{height}"

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

        # ثبت برد برای ماینر
        self.block_counts[winner_node.node_id] += 1

        block = self._construct_block(index, prev_hash, tx_root, winner_node.miner_pk, winner_nonce, winner_pow_hash)
        
        # انتشار بلاک به سایر نودها جهت همگام‌سازی سریع
        for node in self.nodes:
            node.validate_and_add_block(block, vrf_engine=self.vrf_engine, proof_module=self.proof)

    def run(self, num_blocks: int):
        print(f"🎬 Simulation started for {num_blocks} blocks...")
        start_time = time.time()
        
        for h in range(num_blocks):
            self.mine_one_block(h)
            # نمایش پیشرفت کار به ازای هر ۵۰ بلاک
            if (h + 1) % 50 == 0:
                print(f"🔄 Progress: Mined {h + 1}/{num_blocks} blocks...")
                
        print(f"⏱️ Simulation finished in {time.time() - start_time:.2f} seconds.")

    def calculate_fairness_metrics(self):
        counts = list(self.block_counts.values())
        n = len(counts)
        total_blocks = sum(counts)
        mean = total_blocks / n
        
        # ۱. انحراف معیار (Standard Deviation)
        variance = sum((x - mean) ** 2 for x in counts) / n
        std_dev = math.sqrt(variance)
        
        # ۲. نسبت ماکزیمم به مینیمم (Max/Min Ratio)
        max_val = max(counts)
        min_val = min(counts)
        max_min_ratio = max_val / min_val if min_val > 0 else float('inf')
        
        # ۳. شاخص عدالت جین (Jain's Fairness Index) - یک شاخص استاندارد آکادمیک بین 0 و 1
        jains_index = (total_blocks ** 2) / (n * sum(x ** 2 for x in counts)) if total_blocks > 0 else 0

        return {
            "counts": self.block_counts,
            "mean": mean,
            "std_dev": std_dev,
            "max_min_ratio": max_min_ratio,
            "jains_index": jains_index
        }