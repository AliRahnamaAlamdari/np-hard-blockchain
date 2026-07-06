import hashlib, json

class SimpleProof:
    def __init__(self, mode="reveal"):
        self.mode = mode

    def commit_solution(self, solution):
        s = json.dumps(solution, sort_keys=True)
        return hashlib.sha256(s.encode()).hexdigest()

    def verify_solution(self, instance, solution, claimed_cost):
        if instance["type"] == "TSP":
            coords = {c["id"]: c for c in instance["coords"]}
            cost = 0.0
            tour = solution + [solution[0]]
            for i in range(len(tour)-1):
                a = coords[tour[i]]
                b = coords[tour[i+1]]
                cost += ((a["x"]-b["x"])**2 + (a["y"]-b["y"])**2)**0.5
            cost = round(cost, 6)
            return abs(cost - claimed_cost) < 1e-6
        return False
