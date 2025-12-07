import time
import random
import hashlib
import statistics
import matplotlib.pyplot as plt
import pandas as pd


# --- کلاس‌ها و توابع شبیه‌سازی ---
class Miner:
    def __init__(self, id, hash_power, solve_power):
        self.id = id
        self.hash_power = hash_power  # هش بر ثانیه
        self.solve_power = solve_power  # واحد حل مسئله بر ثانیه
        self.reward = 0

def simulate_pow_phase(miners, difficulty):
    """شبیه‌سازی فاز اول: یافتن هش"""
    total_hash_power = sum(m.hash_power for m in miners)
    time_to_find = difficulty / total_hash_power
    # انتخاب برنده بر اساس سهم توان هش
    winner = random.choices(miners, weights=[m.hash_power for m in miners], k=1)[0]
    return winner, time_to_find * random.uniform(0.8, 1.2) # افزودن کمی تصادفی بودن

def simulate_solve_phase(miner, problem_difficulty):
    """شبیه‌سازی فاز دوم: حل مسئله NP-hard"""
    time_to_solve = problem_difficulty / miner.solve_power
    return time_to_solve * random.uniform(0.9, 1.1)

def run_simulation(num_blocks, miners_config, pow_difficulty, np_problem_difficulty):
    """اجرای کامل شبیه‌سازی برای دو سناریو"""
    miners = [Miner(i, hp, sp) for i, (hp, sp) in enumerate(miners_config)]
    
    # --- سناریوی پیشنهادی (Hybrid) ---
    hybrid_results = []
    for _ in range(num_blocks):
        pow_winner, pow_time = simulate_pow_phase(miners, pow_difficulty)
        solve_time = simulate_solve_phase(pow_winner, np_problem_difficulty)
        verification_time = 0.1 # زمان ثابت فرض شده برای تأیید
        total_time = pow_time + solve_time + verification_time
        hybrid_results.append(total_time)
        pow_winner.reward += 1
    
    hybrid_rewards = {m.id: m.reward for m in miners}

    # --- سناریوی پایه (Pure PoW) ---
    # برای مقایسه عادلانه، سختی را طوری تنظیم می‌کنیم که زمان متوسط بلاک مشابه باشد
    baseline_difficulty = pow_difficulty * (statistics.mean(hybrid_results) / (pow_difficulty / sum(m.hash_power for m in miners)))
    baseline_results = []
    for m in miners: m.reward = 0 # Reset rewards
    
    for _ in range(num_blocks):
        _, block_time = simulate_pow_phase(miners, baseline_difficulty)
        baseline_results.append(block_time)

    return hybrid_results, baseline_results, hybrid_rewards

# --- پارامترهای آزمایش ---
NUM_BLOCKS = 200
POW_DIFFICULTY = 10**8
NP_PROBLEM_DIFFICULTY = 5 * 10**6 # سختی مسئله NP-hard

# پروفایل ماینرها: (توان هش, توان حل مسئله)
miners_config = [
    (100, 50), (110, 45), # ماینرهای متوازن
    (500, 10),           # متخصص هش
    (80, 150)            # متخصص حل مسئله
]

# --- اجرا و تحلیل ---
hybrid_times, baseline_times, rewards = run_simulation(NUM_BLOCKS, miners_config, POW_DIFFICULTY, NP_PROBLEM_DIFFICULTY)

# --- رسم نمودار ---
df = pd.DataFrame({
    'Baseline PoW': baseline_times,
    'Hybrid PoW+NP-Hard': hybrid_times
})

plt.figure(figsize=(10, 6))
df.plot(kind='box')
plt.title('Comparison of Block Generation Time')
plt.ylabel('Time (seconds)')
plt.grid(True)
plt.show()

print("--- Analysis ---")
print(f"Average Block Time (Baseline): {statistics.mean(baseline_times):.2f}s")
print(f"Average Block Time (Hybrid): {statistics.mean(hybrid_times):.2f}s")
print(f"Standard Deviation (Baseline): {statistics.stdev(baseline_times):.2f}")
print(f"Standard Deviation (Hybrid): {statistics.stdev(hybrid_times):.2f}")
print("\nReward Distribution in Hybrid Model:")
print(rewards)