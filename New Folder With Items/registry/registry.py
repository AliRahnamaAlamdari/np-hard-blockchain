import json, hashlib

class Registry:
    def __init__(self, json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.leaves = [self._leaf_hash(entry) for entry in self.data["instances"]]
        self.merkle_root = self._build_merkle(self.leaves)
        self.id_to_index = {entry["instance_id"]: i for i, entry in enumerate(self.data["instances"])}

    def size(self):
        return len(self.leaves)

    def _leaf_hash(self, entry):
        s = json.dumps(entry, sort_keys=True)
        return hashlib.sha256(s.encode()).hexdigest()

    def _build_merkle(self, leaves):
        if not leaves:
            return ""
        layer = leaves[:]
        while len(layer) > 1:
            next_layer = []
            for i in range(0, len(layer), 2):
                left = layer[i]
                right = layer[i+1] if i+1 < len(layer) else layer[i]
                next_layer.append(hashlib.sha256((left+right).encode()).hexdigest())
            layer = next_layer
        return layer[0]

    def get_instance_id(self, index):
        return self.data["instances"][index]["instance_id"]

    def get_instance_with_proof(self, index):
        inst = self.data["instances"][index]
        proof = self._merkle_proof(index)
        return inst, proof

    def _merkle_proof(self, index):
        proof = []
        layer = self.leaves[:]
        idx = index
        while len(layer) > 1:
            sibling_idx = idx-1 if idx%2==1 else idx+1
            if sibling_idx < len(layer):
                proof.append(layer[sibling_idx])
            else:
                proof.append(layer[idx])
            next_layer = []
            for i in range(0, len(layer), 2):
                left = layer[i]
                right = layer[i+1] if i+1 < len(layer) else layer[i]
                next_layer.append(hashlib.sha256((left+right).encode()).hexdigest())
            idx = idx//2
            layer = next_layer
        return proof

    def verify_instance_in_manifest(self, instance_id, proof):
        if instance_id not in self.id_to_index:
            return False
        idx = self.id_to_index[instance_id]
        leaf = self.leaves[idx]
        h = leaf
        i = idx
        for sibling in proof:
            if i % 2 == 0:
                h = hashlib.sha256((h+sibling).encode()).hexdigest()
            else:
                h = hashlib.sha256((sibling+h).encode()).hexdigest()
            i = i//2
        return h == self.merkle_root

    def get_instance_by_id(self, instance_id):
        idx = self.id_to_index.get(instance_id)
        if idx is None:
            return None
        return self.data["instances"][idx]
