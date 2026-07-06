import time, hashlib, json

class Block:
    def __init__(self, index, prev_hash, tx_root, nonce, miner_pk, pow_hash, vrf_y=None, vrf_proof=None,
                 repo_id=None, repo_version=None, instance_id=None, merkle_proof=None,
                 solution=None, solution_cost=None, commitment=None, timestamp=None):
        self.index = index
        self.prev_hash = prev_hash
        self.tx_root = tx_root
        self.nonce = nonce
        self.miner_pk = miner_pk
        self.pow_hash = pow_hash
        self.vrf_y = vrf_y
        self.vrf_proof = vrf_proof
        self.repo_id = repo_id
        self.repo_version = repo_version
        self.instance_id = instance_id
        self.merkle_proof = merkle_proof or []
        self.solution = solution
        self.solution_cost = solution_cost
        self.commitment = commitment
        self.timestamp = timestamp or time.time()
        self.hash_value = self.calculate_hash()

    def calculate_hash(self):
        record = json.dumps({
            "index": self.index,
            "prev_hash": self.prev_hash,
            "tx_root": self.tx_root,
            "nonce": self.nonce,
            "miner_pk": self.miner_pk,
            "pow_hash": self.pow_hash,
            "vrf_y": self.vrf_y,
            "instance_id": self.instance_id,
            "commitment": self.commitment,
            "timestamp": self.timestamp
        }, sort_keys=True)
        return hashlib.sha256(record.encode()).hexdigest()

    def to_dict(self):
        return {
            "index": self.index,
            "prev_hash": self.prev_hash,
            "tx_root": self.tx_root,
            "nonce": self.nonce,
            "miner_pk": self.miner_pk,
            "pow_hash": self.pow_hash,
            "vrf_y": self.vrf_y,
            "vrf_proof": self.vrf_proof,
            "repo_id": self.repo_id,
            "repo_version": self.repo_version,
            "instance_id": self.instance_id,
            "merkle_proof": self.merkle_proof,
            "solution": self.solution,
            "solution_cost": self.solution_cost,
            "commitment": self.commitment,
            "timestamp": self.timestamp,
            "hash_value": self.hash_value
        }
