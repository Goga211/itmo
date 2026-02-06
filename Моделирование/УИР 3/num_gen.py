import numpy as np

# === НАСТРОЙКИ ===
N = 1_200_000        # сколько чисел в файле (с запасом под START = 1 000 000)
mean_ta = 183.19    # средний интервал между заявками

# === ГЕНЕРАЦИЯ ТРАССЫ ===
trace = np.random.exponential(scale=mean_ta, size=N)

# === СОХРАНЕНИЕ В ФАЙЛ ===
with open("numbers.txt", "w") as f:
    for x in trace:
        f.write(f"{x:.6f}\n")

print("numbers.txt успешно сгенерирован")
print("Среднее:", trace.mean())
print("СКО:", trace.std())
