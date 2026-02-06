import matplotlib.pyplot as plt

# ДАННЫЕ: (hits, misses) для каждого запуска
runs = [
    {"label": "run 1", "hits": 43612, "misses": 11006},
    {"label": "run 2", "hits": 2047,  "misses": 1},
]

def pie_hit_rate(hits: int, misses: int, title: str):
    total = hits + misses
    hitrate = (hits / total * 100.0) if total else 0.0
    missrate = 100.0 - hitrate

    sizes = [hits, misses]
    labels = ["Попадания", "Промахи"]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.pie(
        sizes,
        labels=labels,
        autopct="%.1f%%",
        startangle=0,
        textprops={"fontsize": 14},
    )
    ax.axis("equal")
    ax.set_title(f"Коэффициент попаданий кеша: {hitrate:.2f}%\n{title}", fontsize=16)

# Рисуем 2 отдельных pie-графика (как на твоём примере)
for r in runs:
    pie_hit_rate(
        r["hits"],
        r["misses"],
        title=f"VTPC | {r['label']}"
    )

plt.show()
