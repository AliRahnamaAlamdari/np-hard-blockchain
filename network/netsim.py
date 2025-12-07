# simulator.py

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
    def __init__(self, num_nodes: int, difficulty: int, repo: Registry, mode: str = "pow-np"):
        """
        :param mode: 'pow' for pure PoW, 'pow-np' for proposed PoW + NP-Hard solution.
        """
        self.repo = repo
        self.difficulty = difficulty
        self.mode = mode  # حالت شبیه سازی
        self.vrf_engine = SimpleVRF()
        self.solver = TSPSolver()
        self.proof = SimpleProof(mode="reveal")
        self.nodes: List[Node] = [
            Node(node_id=i, difficulty=difficulty, repo=repo) for i in range(num_nodes)
        ]
        self.log_data: List[Dict[str, Any]] = [] # برای جمع آوری داده های ارزیابی

    def _construct_block(self, index: int, prev_hash: str, tx_root: str,
                         miner_pk: str, nonce: int, pow_hash: str) -> Tuple[Block, float]:
        """
        ساخت بلاک و اجرای مرحله VRF/NP-Hard اگر حالت 'pow-np' باشد.
        زمان صرف شده برای حل NP-Hard را برمی گرداند.
        """
        vrf_y = vrf_proof = None
        instance_id = merkle_proof = instance = None
        solution = solution_cost = commitment = None
        solve_time = 0.0

       
        if self.mode == "pow-np" and self.vrf_engine and self.repo and self.solver and self.proof:
            
            start_solve = time.time() 
            
            # 1. VRF stage
            vrf_y, vrf_proof = self.vrf_engine.prove(pow_hash)
            vrf_int = int(vrf_y, 16)
            n = self.repo.size()
            
            # 2. Sample instance
            idx = vrf_int % n
            instance_id = self.repo.get_instance_id(idx)
            instance, merkle_proof = self.repo.get_instance_with_proof(idx)
            
            # 3. Solve instance
            solution, solution_cost = self.solver.solve(instance)
            
            # 4. Commit solution
            commitment = self.proof.commit_solution(solution)

            end_solve = time.time() 
            solve_time = end_solve - start_solve


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
        return block, solve_time

    def mine_one_block(self, height: int):
        log = BlockEventLog(height)
     
        tip = self.nodes[0].blockchain.get_last_block()
        log.add("Mode", self.mode)
        log.add("Tip Hash", tip.hash_value[:16])
        log.add("Difficulty", self.difficulty)
        log.add("Total Miners", len(self.nodes))
        index = tip.index + 1
        prev_hash = tip.hash_value

        tx_root = f"txroot-{height}"

        winner_node = None
        winner_nonce = None
        winner_pow_hash = None

        nonce = 0
        print(f"\n=== Mining height {index} | Mode: {self.mode} | Difficulty: {self.difficulty} ===")
        
        start_mine = time.time() 

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
            
        end_pow = time.time() # ⭐️ پایان زمان PoW (یافتن هش)
        pow_time = end_pow - start_mine

        # ⭐️ ساخت بلوک و انجام مرحله حل مسئله در صورت لزوم
        block, solve_time = self._construct_block(index, prev_hash, tx_root, winner_node.miner_pk, winner_nonce, winner_pow_hash)

        end_mine = time.time() # ⭐️ پایان زمان کل تولید بلوک (استخراج + حل)
        total_block_time = end_mine - start_mine

        # ⭐️ جمع آوری داده های ارزیابی
        self.log_data.append({
            "height": index,
            "mode": self.mode,
            "difficulty": self.difficulty,
            "nonce_count": winner_nonce,
            "pow_time": pow_time,
            "solve_time": solve_time,
            "total_block_time": total_block_time,
            "solution_cost": block.solution_cost
        })
        print(f"⌛ PoW Time: {pow_time:.4f}s, Solve Time: {solve_time:.4f}s, Total Time: {total_block_time:.4f}s")


        # broadcast
        for node in self.nodes:
            # توجه: در اینجا ما بلوک را به زنجیره نودها اضافه می کنیم، اما برای شبیه سازی PoW خالص،
            # نیازی به VRF/NP-Hard نیست، اما چون `validate_block` در `chain.py` این بخش ها را نادیده می گیرد
            # در صورتی که `vrf_engine` یا `repo` در خود بلوک نباشد، این بخش ها باعث خطا نمی شوند.
            # برای اطمینان بیشتر، در حالت 'pow'، فیلدهای VRF/NP-Hard در خود بلوک خالی خواهند بود.
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
            print(f"\n🧩 Node {node.node_id} chain (Mode: {self.mode}):")
            for blk in node.blockchain.chain:
                # نمایش زمان حل فقط اگر وجود داشته باشد
                cost_str = f"| cost={blk.solution_cost}" if blk.solution_cost is not None else ""
                print(
                    f"  ▪ Height {blk.index} | Hash {blk.hash_value[:12]}... | Miner {blk.miner_pk} {cost_str}"
                )
            last_hashes.add(node.blockchain.get_last_block().hash_value)

        print("\n------------------------------")
        if len(last_hashes) == 1:
            print(f"✅ All nodes synchronized — same tip hash in {self.mode} mode.")
        else:
            print(f"⚠️ Chains diverged in {self.mode} mode!")