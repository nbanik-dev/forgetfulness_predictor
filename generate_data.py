import numpy as np
import pandas as pd

np.random.seed(42)
N = 1000

time_since_study = np.random.randint(0, 31, N)           # days (0–30)
revisions        = np.random.randint(0, 11, N)           # 0–10
study_duration   = np.round(np.random.uniform(0.5, 8, N), 1)  # hours
difficulty       = np.random.choice(["Easy", "Medium", "Hard"], N)
cgpa             = np.round(np.random.uniform(2.0, 4.0, N), 2)

diff_map = {"Easy": 1, "Medium": 2, "Hard": 3}
diff_num = np.array([diff_map[d] for d in difficulty])

# Ebbinghaus-inspired retention formula
stability = 1 + revisions * 0.6 + study_duration * 0.3
base = 100 * np.exp(-0.35 * time_since_study / stability)

# Adjustments
retention = (
    base
    - (diff_num - 1) * 4          # harder topics → lower retention
    + (cgpa - 2.0) * 4            # higher CGPA → better retention
    + np.random.normal(0, 4, N)   # noise
)
retention = np.clip(retention, 0, 100).round(2)

df = pd.DataFrame({
    "Time_since_study": time_since_study,
    "Revisions":        revisions,
    "Study_duration":   study_duration,
    "Difficulty":       difficulty,
    "CGPA":             cgpa,
    "Retention":        retention,
})

df.to_csv("data.csv", index=False)
print(f"✅ data.csv created with {len(df)} rows.")
