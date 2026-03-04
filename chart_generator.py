import pandas as pd
import matplotlib.pyplot as plt

# Load your CSV
df = pd.read_csv("benchmark_results.csv")

# Plot average times
fig, ax = plt.subplots(figsize=(10,6))
strategies = ["Adaptive_Avg(s)", "s1_Avg(s)", "s7_Avg(s)"]

df.plot(x="Dataset", y=strategies, kind="bar", ax=ax)

ax.set_ylabel("Execution Time (s)")
ax.set_title("WeTriC Spreading Benchmark Results")
plt.xticks(rotation=45)
plt.legend(["Adaptive", "Spread=1", "Spread=7"])
plt.tight_layout()
plt.show()