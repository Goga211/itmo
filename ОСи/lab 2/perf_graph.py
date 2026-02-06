import os
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "summary_perf.csv"     # <-- поправь если у тебя другое имя
OUT_DIR = "plots_perf_off_random"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

# фильтр: direct=off, random
df = df[(df["direct"] == "off") & (df["type"] == "random")].copy()

# только perf-метрики (чисто из perf)
METRICS = {
    "cycles": "CPU cycles",
    "instructions": "Instructions",
    "ipc": "IPC (instructions/cycle)",
    "cache_refs": "Cache references",
    "cache_miss": "Cache misses",
    "cache_miss_pct": "Cache miss rate (%)",
    "branches": "Branches",
    "branch_misses": "Branch misses",
    "branch_miss_pct": "Branch miss rate (%)",
    "page_faults": "Page faults",
}

# проверка что все колонки есть
missing = [c for c in METRICS.keys() if c not in df.columns]
if missing:
    print("❌ Не хватает колонок в CSV:", missing)
    print("Колонки которые есть:", list(df.columns))
    raise SystemExit(1)

block_sizes = sorted(df["block_size"].unique())

for bs in block_sizes:
    df_bs = df[df["block_size"] == bs].copy()

    # для удобства сортируем по размеру файла
    df_bs = df_bs.sort_values("file_size_mb")

    for metric, ylabel in METRICS.items():
        # pivot: X=file_size_mb, линии=backend(os/vtpc)
        piv = df_bs.pivot_table(
            index="file_size_mb",
            columns="backend",
            values=metric,
            aggfunc="mean"
        ).sort_index()

        # если данных нет — пропускаем
        if piv.empty:
            continue

        plt.figure()
        for backend in piv.columns:
            plt.plot(piv.index, piv[backend], marker="o", label=str(backend))

        plt.title(f"{metric} vs file_size_mb | direct=off random | block_size={bs}")
        plt.xlabel("file_size_mb")
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.legend()

        out_path = os.path.join(OUT_DIR, f"{metric}_bs{bs}.png")
        plt.savefig(out_path, dpi=160, bbox_inches="tight")
        plt.close()

print(f"✅ Готово! Графики сохранены в папку: {OUT_DIR}/")
