import sys, glob, pandas as pd, matplotlib.pyplot as plt, os, re

def load(paths):
    dfs = []
    for p in paths:
        df = pd.read_csv(p)
        df["model"] = df["model"].fillna("unknown")
        df["conc"]  = int(re.search(r'_c(\d+)\.csv$', p).group(1)) if re.search(r'_c(\d+)\.csv$', p) else 1
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def agg(df, metric):
    g = df[df["ok"]==True].groupby(["model","conc"])[metric]
    return g.mean().reset_index().pivot(index="conc", columns="model", values=metric)

paths = sys.argv[1:]
df = load(paths)
os.makedirs("plots", exist_ok=True)

for metric in ["ttft_ms","e2e_ms","tps","tpot_ms","itl_mean_ms","itl_p95_ms"]:
    piv = agg(df, metric)
    ax = piv.plot(title=f"{metric} vs concurrency")
    plt.xlabel("concurrency"); plt.ylabel(metric); plt.tight_layout(); plt.savefig(f"plots/{metric}_vs_conc.png"); plt.close()

# RPS/System TPS (derive from per-file summaries by recomputing here)
def file_level_stats(path):
    d = pd.read_csv(path)
    # We don't have wall time per file; infer concurrency from filename (_cX) and assume duration from test settings if desired.
    # Simpler: approximate RPS as len(ok)/duration if you supply duration; for plotting, use e2e_ms inversely.
    return None