import hashlib, json

def merkle_root_from_instances(instances):
    leaves = [hashlib.sha256(json.dumps(i, sort_keys=True).encode()).hexdigest() for i in instances]
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
