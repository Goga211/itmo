import random
import math
import numpy as np
import matplotlib.pyplot as plt

sample_size = 300
k = 3
l = 0.01638

def erlang_generator(k, l):
    result = 0.0
    for i in range(k):
        u = random.uniform(0, 1)
        exp = - (1/l) * math.log(u)
        result += exp
    return result

erlang_samples = [erlang_generator(k, l) for _ in range(sample_size)]
x = np.array(erlang_samples)

bins = 18
counts, edges, _ = plt.hist(x, bins=bins, edgecolor='black',
                            alpha=0.7, label="Гистограмма данных", color='steelblue')

x_vals = np.linspace(min(x), max(x), 400)
factorial = math.factorial(k - 1)
y_vals = (l**k) * (x_vals**(k - 1)) * np.exp(-l * x_vals) / factorial
scale = len(x) * (edges[1] - edges[0])
y_vals_scaled = y_vals * scale

plt.plot(x_vals, y_vals_scaled, 'r-', linewidth=2,
         label=f'Распределение Эрланга (k={k}, λ={l:.4f})')

plt.title("Гистограмма распределения частот с аппроксимацией Эрланга")
plt.xlabel("Интервалы значений")
plt.ylabel("Частота")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.tight_layout()
plt.show()


outfile = "erlang_samples.csv"
rounded = np.round(x, 3)
np.savetxt(
    outfile,
    rounded,
    delimiter=",",
    fmt="%.3f",
    header="erlang_sample",
    comments=""
)
print(f"Сохранено {len(rounded)} значений в {outfile}")