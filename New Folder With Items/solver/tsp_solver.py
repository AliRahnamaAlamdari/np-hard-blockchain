import itertools, math

def euclidean(a,b):
    return math.hypot(a["x"]-b["x"], a["y"]-b["y"])

class TSPSolver:
    def __init__(self, time_budget=None):
        self.time_budget = time_budget

    def solve(self, instance):
        coords = instance["coords"]
        n = len(coords)
        best = None
        best_cost = float("inf")
        ids = [c["id"] for c in coords]
        pos = {c["id"]: c for c in coords}
        start = ids[0]
        for perm in itertools.permutations(ids[1:]):
            tour = [start] + list(perm) + [start]
            cost = 0.0
            for i in range(len(tour)-1):
                cost += euclidean(pos[tour[i]], pos[tour[i+1]])
            if cost < best_cost:
                best_cost = cost
                best = tour[:-1]
        return best, round(best_cost, 6)
