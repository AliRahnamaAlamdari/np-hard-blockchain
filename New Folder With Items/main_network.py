
import os
from registry.registry import Registry
from network.simulator import NetworkSimulator

def main():
    base_dir = os.path.dirname(__file__)
    repo_path = os.path.join(base_dir, "registry", "registry.json")
    repo = Registry(repo_path)
    difficulty = 4  # می‌توانی سختی را اینجا تنظیم کنی
    sim = NetworkSimulator(num_nodes=8, difficulty=difficulty, repo=repo)
    sim.run(num_blocks=3)
    sim.print_chains()

if __name__ == "__main__":
    main()
