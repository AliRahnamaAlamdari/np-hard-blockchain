
import os
from problem_size_sim import ProblemSizeSimulator
from registry.registry import Registry
from network.simulator import NetworkSimulator
from fairness import FairnessNetworkSimulator

# def main():
#     base_dir = os.path.dirname(__file__)
#     repo_path = os.path.join(base_dir, "registry", "registry.json")
#     repo = Registry(repo_path)
#     difficulty = 6  
#     sim = NetworkSimulator(num_nodes=8, difficulty=difficulty, repo=repo)
#     sim.run(num_blocks=6)
#     sim.print_chains()

# def main():
#     base_dir = os.path.dirname(__file__)
#     repo_path = os.path.join(base_dir, "registry", "registry.json")
#     repo = Registry(repo_path)
    
#     # Configuration for difficulty levels
#     difficulty_levels = {
#         "Low": 4,
#         "Medium": 5,
#         "High": 6
#     }
    
#     num_blocks_to_mine = 6
#     report_data = {}

#     print("=== Starting Time Share Measurement Experiment (PoW vs. Useful Work) ===")

#     for level_name, diff_value in difficulty_levels.items():
#         print(f"\n--- Running Network Simulator with {level_name} Difficulty ---")
        
#         sim = NetworkSimulator(num_nodes=8, difficulty=diff_value, repo=repo)
#         sim.run(num_blocks=num_blocks_to_mine)
        
#         # Calculate percentages
#         t_pow = sim.total_pow_time
#         t_work = sim.total_work_time
#         t_total = t_pow + t_work
        
#         pow_percentage = (t_pow / t_total) * 100 if t_total > 0 else 0
#         work_percentage = (t_work / t_total) * 100 if t_total > 0 else 0
        
#         report_data[level_name] = {
#             "diff": diff_value,
#             "pow_time": t_pow,
#             "work_time": t_work,
#             "pow_pct": pow_percentage,
#             "work_pct": work_percentage
#         }

#     # --- Print Final English Report Table ---
#     print("\n" + "="*85)
#     print("📊 FINAL REPORT: Time Share Comparison Based on PoW Difficulty")
#     print("="*85)
#     print(f"{'Difficulty Level':<18} | {'PoW Time (s)':<14} | {'Useful Work (s)':<17} | {'PoW Share':<11} | {'Useful Work Share'}")
#     print("-"*85)
    
#     for level_name, metrics in report_data.items():
#         print(f"{level_name:<18} | "
#               f"{metrics['pow_time']:<14.4f} | "
#               f"{metrics['work_time']:<17.4f} | "
#               f"{metrics['pow_pct']:>10.2f}% | "
#               f"{metrics['work_pct']:>18.2f}%")
        
#     print("="*85)
#     print("💡 Analysis: As PoW difficulty increases, the time share of Useful Work (TSP) drops drastically.\n"
#           "   The network shifts from being productive (solving problems) to wasting energy on hashing.")


# def main():
#     base_dir = os.path.dirname(__file__)
#     repo_path = os.path.join(base_dir, "registry", "registry.json")
#     repo = Registry(repo_path)
    
#     # تنظیمات آزمایش عدالت
#     NUM_MINERS = 5
#     NUM_BLOCKS = 500
#     DIFFICULTY = 2  # Keep it low for fast execution of 500 blocks
    
#     print(f"=== Mining Fairness Experiment ===")
#     print(f"Parameters: Miners = {NUM_MINERS}, Total Blocks = {NUM_BLOCKS}, PoW Difficulty = {DIFFICULTY}")
#     print("==================================")
    
#     # اجرای شبیه‌ساز
#     sim = FairnessNetworkSimulator(num_nodes=NUM_MINERS, difficulty=DIFFICULTY, repo=repo)
#     sim.run(num_blocks=NUM_BLOCKS)
    
#     # محاسبه شاخص‌ها
#     metrics = sim.calculate_fairness_metrics()
    
#     # --- چاپ گزارش نهایی به انگلیسی ---
#     print("\n" + "="*50)
#     print("📊 FAIRNESS EXPERIMENT REPORT")
#     print("="*50)
#     print(f"{'Miner ID':<12} | {'Blocks Won':<12} | {'Percentage':<10}")
#     print("-"*50)
    
#     for miner_id, blocks in metrics["counts"].items():
#         percentage = (blocks / NUM_BLOCKS) * 100
#         print(f"Node {miner_id:<7} | {blocks:<12} | {percentage:>8.2f}%")
        
#     print("-"*50)
#     print(f"💡 Expected Blocks per Miner (Mean): {metrics['mean']:.2f}")
#     print(f"📉 Standard Deviation (SD)        : {metrics['std_dev']:.4f}  (Lower is better)")
#     print(f"📈 Max/Min Block Ratio            : {metrics['max_min_ratio']:.4f}  (Closer to 1.0 is better)")
#     print(f"⚖️ Jain's Fairness Index          : {metrics['jains_index']:.4f}  (Closer to 1.0 is better)")
#     print("="*50)

# def main():
#     base_dir = os.path.dirname(__file__)
#     repo_path = os.path.join(base_dir, "registry", "registry.json")
#     repo = Registry(repo_path)
    
#     # Setup configuration
#     NUM_NODES = 6
#     DIFFICULTY = 3
#     NUM_BLOCKS = 5
    
#     sim = NetworkSimulator(num_nodes=NUM_NODES, difficulty=DIFFICULTY, repo=repo)
    
#     print(f"=== Starting Validation Profiling Experiment ===")
#     print(f"Nodes: {NUM_NODES} | Difficulty: {DIFFICULTY} | Blocks to mine: {NUM_BLOCKS}")
#     print("Running simulation...")
    
#     sim.run(num_blocks=NUM_BLOCKS)
    
#     # Calculate Averages
#     v_count = sim.total_validations_count
#     avg_pow_check = (sim.pow_check_time / v_count) * 1000       # Convert to milliseconds
#     avg_vrf_check = (sim.vrf_check_time / v_count) * 1000
#     avg_merkle_check = (sim.merkle_check_time / v_count) * 1000
#     avg_sol_check = (sim.solution_check_time / v_count) * 1000
    
#     avg_miner_solve = (sim.total_mining_solve_time / sim.num_blocks_mined) * 1000 if sim.num_blocks_mined > 0 else 0

#     # --- PRINT DETAILED VALIDATION REPORT ---
#     print("\n" + "="*70)
#     print("📊 DETAILED VALIDATION STEPS PERFORMANCE REPORT")
#     print("="*70)
#     print(f"Total validation actions profiled: {v_count} (Blocks * Nodes)")
#     print("-"*70)
#     print(f"{'Validation Check Step':<40} | {'Average Latency (ms)':<20}")
#     print("-"*70)
#     print(f"1. PoW Check (Hash & Difficulty)        | {avg_pow_check:<20.6f} ms")
#     print(f"2. VRF Proof Verification                | {avg_vrf_check:<20.6f} ms")
#     print(f"3. Merkle Proof Verification             | {avg_merkle_check:<20.6f} ms")
#     print(f"4. Solution Feasibility & Cost Check    | {avg_sol_check:<20.6f} ms")
#     print("="*70)
    
#     # --- PROOF OF NON-RE-SOLVING (The Core Proof) ---
#     print("\n" + "="*70)
#     print("💡 PROOF OF COMPLEXITY ASYMMETRY (Miner Solving vs. Validator Checking)")
#     print("="*70)
#     print(f"🧠 Miner Avg Time to SOLVE TSP (NP-Hard)  : {avg_miner_solve:.4f} ms")
#     print(f"🔎 Validator Avg Time to CHECK TSP (O(N)) : {avg_sol_check:.4f} ms")
    
#     if avg_sol_check > 0:
#         speedup = avg_miner_solve / avg_sol_check
#         print(f"🚀 Verification is {speedup:.1f}x FASTER than solving from scratch!")
#     print("="*70)
#     print("Conclusion: The Validator DOES NOT re-solve the TSP problem. It acts as a lightweight\n"
#           "verifier using mathematical/cryptographic shortcuts, keeping the blockchain scalable.")
# if __name__ == "__main__":
#     main()



def main():
    base_dir = os.path.dirname(__file__)
    repo_path = os.path.join(base_dir, "registry", "registry.json")
    repo = Registry(repo_path)

    # Fixed PoW difficulty for a fair test
    FIXED_DIFFICULTY = 3 
    BLOCKS_PER_TEST = 3
    
    simulator = ProblemSizeSimulator(difficulty=FIXED_DIFFICULTY, repo=repo)
    report_results = []

    print("=== Starting NP-Hard Problem Size Scaling Experiment ===")
    print(f"Fixed PoW Difficulty: {FIXED_DIFFICULTY} | Blocks per instance: {BLOCKS_PER_TEST}")
    print("Evaluating instances from registry...")

    # Sort instances by size N to see the progression
    sorted_instances = sorted(repo.data["instances"], key=lambda x: x["n"])

    for instance in sorted_instances:
        n_size = instance["n"]
        instance_id = instance["instance_id"]
        
        # Skip overly large instances if you suspect your solver is Brute Force (O(N!))
        if n_size > 15: 
            continue
            
        print(f" -> Testing Instance {instance_id} (N = {n_size})...")
        avg_pow, avg_work = simulator.run_benchmark_for_instance(instance, num_blocks=BLOCKS_PER_TEST)
        
        total = avg_pow + avg_work
        pow_pct = (avg_pow / total) * 100
        work_pct = (avg_work / total) * 100
        
        report_results.append({
            "n": n_size,
            "id": instance_id,
            "pow_time": avg_pow,
            "work_time": avg_work,
            "pow_pct": pow_pct,
            "work_pct": work_pct
        })

    # --- PRINT FINAL REPORT TABLE ---
    print("\n" + "="*85)
    print("📊 IMPACT OF NP-HARD PROBLEM SIZE (N) ON COMPUTATION SHARE")
    print("="*85)
    print(f"{'Instance ID':<12} | {'Size (N)':<8} | {'Avg PoW (s)':<12} | {'Avg Useful Work (s)':<20} | {'Useful Work %'}")
    print("-"*85)
    
    for res in report_results:
        print(f"{res['id']:<12} | "
              f"{res['n']:<8} | "
              f"{res['pow_time']:<12.5f} | "
              f"{res['work_time']:<20.5f} | "
              f"{res['work_pct']:>12.2f}%")
        
    print("="*85)
    print("💡 Key Takeaway: As N grows linearly, Useful Work execution time grows exponentially.\n"
          "   Consequently, the network shifts from wasting energy on hashing to spending almost\n"
          "   all its computational resources on solving the actual economic/mathematical problem.")

if __name__ == "__main__":
    main()