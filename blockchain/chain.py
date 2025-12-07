from blockchain.block import Block
import time, hashlib

class Blockchain:
    def __init__(self, difficulty=3):
        self.chain = [self.create_genesis_block()]
        self.difficulty = difficulty

    def create_genesis_block(self):
        return Block(0, "0", "genesis", 0, "genesis", "0"*64, timestamp=time.time())

    def get_last_block(self):
        return self.chain[-1]

    def calculate_pow_hash(self, index, prev_hash, tx_root, nonce, miner_pk, vrf_y=""):
        record = f"{index}{prev_hash}{tx_root}{nonce}{miner_pk}{vrf_y}"
        return hashlib.sha256(record.encode()).hexdigest()

    def mine_block(self, miner_pk, tx_root="txroot", repo_id=None, repo_version=None,
                   vrf_engine=None, repo=None, solver=None, proof_module=None):
        index = len(self.chain)
        prev_hash = self.get_last_block().hash_value
        nonce = 0
        print(f"⛏ Mining block {index}... (difficulty={self.difficulty})")
        while True:
            pow_hash = self.calculate_pow_hash(index, prev_hash, tx_root, nonce, miner_pk)
            if pow_hash.startswith("0"*self.difficulty):
                print(f"✅ PoW success: {pow_hash[:16]}... nonce={nonce}")
                # VRF stage (if engine provided)
                vrf_y = vrf_proof = None
                instance_id = merkle_proof = instance = None
                solution = None
                solution_cost = None
                commitment = None
                if vrf_engine and repo:
                    vrf_y, vrf_proof = vrf_engine.prove(pow_hash)
                    # sample index
                    idx = int(vrf_y, 16) % repo.size()
                    instance_id = repo.get_instance_id(idx)
                    instance, merkle_proof = repo.get_instance_with_proof(idx)
                    # solve using solver
                    if solver:
                        solution, solution_cost = solver.solve(instance)
                        # produce commitment / proof (prototype: commit by hash; real SNARK omitted)
                        commitment = proof_module.commit_solution(solution) if proof_module else None
                new_block = Block(index, prev_hash, tx_root, nonce, miner_pk, pow_hash,
                                  vrf_y, vrf_proof, repo_id, repo_version,
                                  instance_id, merkle_proof, solution, solution_cost, commitment)
                self.chain.append(new_block)
                return new_block
            nonce += 1

    def validate_block(self, block, vrf_engine=None, repo=None, proof_module=None):
        # Validate PoW
        expected_pow = self.calculate_pow_hash(block.index, block.prev_hash, block.tx_root, block.nonce, block.miner_pk, "")
        if expected_pow != block.pow_hash:
            return False, "Invalid PoW hash mismatch"
        if not block.pow_hash.startswith("0"*self.difficulty):
            return False, "Pow does not meet difficulty"
        # VRF verification (if present)
        if vrf_engine and block.vrf_y and block.vrf_proof:
            ok, recomputed = vrf_engine.verify(block.pow_hash, block.vrf_proof)
            if not ok or recomputed != block.vrf_y:
                return False, "Invalid VRF proof"
            # check sampled index matches instance_id via repo merkle proof if repo present
            if repo and block.instance_id:
                if not repo.verify_instance_in_manifest(block.instance_id, block.merkle_proof):
                    return False, "Instance merkle inclusion failed"
        # Verify solution (prototype: solution revealed; verify by recomputing cost)
        if block.solution is not None and repo:
            inst = repo.get_instance_by_id(block.instance_id)
            if not inst:
                return False, "Instance not found for verification"
            if not proof_module.verify_solution(inst, block.solution, block.solution_cost):
                return False, "Solution verification failed"
        return True, "OK"
