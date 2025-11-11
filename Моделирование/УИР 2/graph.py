import matplotlib.pyplot as plt
import pandas as pd

metrics = {
    "Нагрузка (y)":                (6.400000, 6.400000),
    "Загрузка ср. (ρ̄)":          (0.826019, 0.761215),
    "Длина очереди (Lq)":         (3.096581, 0.294053),
    "Число заявок (L)":           (4.748621, 1.816483),
    "Время ожидания (Wq, c)":     (14.995207, 1.545180),
    "Время пребывания (W, c)":    (22.995218, 9.545197),
    "Вероятность потери (π_loss)":(0.741869, 0.762121),
    "Производительность (λ′)":    (0.206505, 0.190303),
}

df = pd.DataFrame(metrics, index=["Система 1","Система 2"]).T

ax = df.plot(kind="bar")
ax.set_ylabel("Значение")
ax.set_title("Сравнение характеристик: Система 1 vs Система 2")
ax.legend(title="Системы", loc="best", frameon=False)
plt.xticks(rotation=30, ha="right")

for container in ax.containers:
    ax.bar_label(container, fmt="%.3f", padding=3, rotation=90)

plt.tight_layout()
plt.savefig("systems_comparison_labeled.png", dpi=200)
plt.show()
