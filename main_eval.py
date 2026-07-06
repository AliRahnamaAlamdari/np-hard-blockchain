# main.py
import os
from registry.registry import Registry
from network.netsim import NetworkSimulator

def run_simulation(mode, difficulty, num_blocks, repo):
    print(f"\n==========================================")
    print(f"🚀 Running Simulation: {mode} (Difficulty: {difficulty})")
    print(f"==========================================")
    sim = NetworkSimulator(num_nodes=8, difficulty=difficulty, repo=repo, mode=mode)
    sim.run(num_blocks=num_blocks)
    sim.print_chains()
    return sim.log_data

def main():
    base_dir = os.path.dirname(__file__)
    repo_path = os.path.join(base_dir, "registry", "registry.json")
    repo = Registry(repo_path)
    
    num_blocks = 10 # تعداد بلوک ها را برای آماری بودن افزایش دهید

    # 1. سناریوی PoW پایه (سختی بالا)
    pow_difficulty = 6 # سختی بالاتری که تقریبا زمان حل آن با PoW-NP برابر شود
    pow_data = run_simulation("pow", pow_difficulty, num_blocks, repo)
    
    # 2. سناریوی PoW-NP (مدل پیشنهادی - سختی پایین)
    np_difficulty = 4 # سختی پایین (همان 4 قبلی)
    np_data = run_simulation("pow-np", np_difficulty, num_blocks, repo)
    
    # 3. تحلیل و گزارش نتایج
    analyze_results(pow_data, np_data)


def analyze_results(pow_data, np_data):
    # تحلیل داده ها و چاپ نتایج میانگین
    def get_avg(data, key):
        return sum(d[key] for d in data) / len(data) if data else 0

    print("\n==========================================")
    print("✨ Comparison Results ✨")
    print("==========================================")

    # مقایسه زمان تولید بلوک
    avg_pow_time_pow = get_avg(pow_data, "total_block_time")
    avg_pow_time_np = get_avg(np_data, "total_block_time")
    print(f"Total Block Time (PoW {pow_data[0]['difficulty']}): {avg_pow_time_pow:.4f}s")
    print(f"Total Block Time (PoW-NP {np_data[0]['difficulty']}): {avg_pow_time_np:.4f}s")
    
    # مقایسه هزینه محاسباتی PoW (تعداد Nonce)
    avg_nonce_pow = get_avg(pow_data, "nonce_count")
    avg_nonce_np = get_avg(np_data, "nonce_count")
    print(f"Average PoW Nonces (PoW): {avg_nonce_pow:.0f}")
    print(f"Average PoW Nonces (PoW-NP): {avg_nonce_np:.0f}")
    
    # هزینه حل مسئله NP-Hard
    avg_solve_time = get_avg(np_data, "solve_time")
    print(f"Average NP-Hard Solve Time (PoW-NP): {avg_solve_time:.4f}s")
    
    # مقایسه زمان PoW تنها
    avg_pow_only_time_pow = get_avg(pow_data, "pow_time")
    avg_pow_only_time_np = get_avg(np_data, "pow_time")
    print(f"Average PoW Hash Time (PoW): {avg_pow_only_time_pow:.4f}s")
    print(f"Average PoW Hash Time (PoW-NP): {avg_pow_only_time_np:.4f}s")

    # 

if __name__ == "__main__":
    main()