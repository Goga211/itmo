import matplotlib.pyplot as plt
import numpy as np

variants = [f"Вариант {i}" for i in range(1, 10)]
x = np.arange(len(variants))


# ======== ОБНОВЛЁННЫЕ ДАННЫЕ ========

prob_loss = [
    0.041,   # Вариант 1
    0.120,   # Вариант 2
    0.207,   # Вариант 3
    0.042,   # Вариант 4 (← заменён с варианта 6)
    0.112,   # Вариант 5
    0.203,   # Вариант 6 (← заменён с варианта 4)
    0.0120,  # Вариант 7
    0.0633,  # Вариант 8
    0.1426   # Вариант 9
]

queue_len = [
    0.171,
    0.402,
    0.605,
    0.155,   # Вариант 4
    0.396,
    0.605,   # Вариант 6
    0.110,
    0.366,
    0.635
]

load = [
    0.476,
    0.655,
    0.752,
    0.454,   # Вариант 4
    0.649,
    0.742,   # Вариант 6
    0.498,
    0.686,
    0.804
]

wait_time = [
    32.828,
    149.898,
    139.84,
    30.6,       # Вариант 4
    82.857,
    135.521,    # Вариант 6
    20.301,
    72.652,
    137.488
]

transient_len = [
    10000,
    10000,
    5000,
    10000,   # Вариант 4 теперь имеет T=10000
    5000,
    5000,    # Вариант 6 теперь имеет T=5000
    50000,
    10000,
    10000
]


# ===================================

def plot_bar(values, title, ylabel, percent=False):
    plt.figure(figsize=(12, 6))
    bars = plt.bar(x, values, color="#4a90e2")

    for bar, val in zip(bars, values):
        label = f"{val*100:.2f}%" if percent else f"{val}"
        plt.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height(),
                 label,
                 ha='center', va='bottom', fontsize=11)

    plt.xticks(x, variants)
    plt.title(title, fontsize=20)
    plt.ylabel(ylabel, fontsize=15)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()


# === Построение всех графиков ===

plot_bar(prob_loss, "Вероятность потери", "Вероятность", percent=True)
plot_bar(queue_len, "Длина очереди", "Длина очереди")
plot_bar(load, "Загрузка системы", "Загрузка", percent=True)
plot_bar(wait_time, "Среднее время ожидания", "Время ожидания")
plot_bar(transient_len, "Длина переходного процесса", "Число заявок")
