"""Generate a synthetic headerless CLAP-style export (no real data) for smoke tests."""
import random
import sys

import pandas as pd

random.seed(1)
n = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
rows = []
for i in range(n):
    clustered = i % 10 == 0  # one surveyor stamps many records at the same spot
    lat = 29.0 + (0.0 if clustered else random.random() * 0.5)
    lng = 76.0 + (0.0 if clustered else random.random() * 0.5)
    a, l = random.randint(10, 100), random.randint(5, 100)
    rows.append([f"{1000000000 + i}", f"APP{i}", f"Farmer {i}", "Flood", "2026-09-01", "2026-09-05", 1.2, "Paddy",
                 str(random.randint(1, 60)), "0", "StateA", f"Dist{i % 3}", "T1", "B1", f"V{i % 20}", "P1",
                 "Surveyor X" if clustered else f"Surveyor {i % 25}", f"9{random.randint(10**8, 10**9 - 1)}",
                 a, l, a * l / 100, "", lat, lng, "", ""])
pd.DataFrame(rows).to_excel(sys.argv[1], header=False, index=False)
