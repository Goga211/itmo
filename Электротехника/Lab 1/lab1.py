import matplotlib.pyplot as plt

# Данные из таблицы
I = [0, 2, 4, 6, 8, 10, 12, 14.0156, 15.9474, 17.9412, 20]   # мА
U = [3, 2.7, 2.4, 2.1, 1.8, 1.5, 1.2, 0.897, 0.606, 0.305, 0]   # В
P = [0, 0.0054, 0.0096, 0.0126, 0.0144, 0.015, 0.0144, 0.0126, 0.0097, 0.0055, 0]  # Вт
eta = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.299, 0.202, 0.102, 0]                     # КПД

# --- 1. Внешняя характеристика источника ---
plt.figure(figsize=(6,4))
plt.plot(I, U, 'o-', label="U(I)", color="blue")
for x, y in zip(I, U):
    plt.annotate(f"{x:.5g}; {y:.5g}", (x, y), textcoords="offset points", xytext=(5,5), fontsize=8)
plt.xlabel("I, мА")
plt.ylabel("U, В")
plt.title("Внешняя характеристика источника")
plt.grid(True)
plt.legend()
plt.show()

# --- 2. Рабочие характеристики ---
# Масштабируем КПД (умножаем на максимум мощности)
eta_scaled = [e * max(P) for e in eta]  # η × 0.015

plt.figure(figsize=(6,4))
plt.plot(I, P, 'o-', label="P(I)", color="blue")
plt.plot(I, eta_scaled, 's-', label="η", color="orange")

# подписи для P(I)
for x, y in zip(I, P):
    plt.annotate(f"{y:.3g}", (x, y), textcoords="offset points", xytext=(5,5), fontsize=8, color="blue")

# подписи для η(I)
for x, y in zip(I, eta):
    plt.annotate(f"{y:.2g}", (x, y), textcoords="offset points", xytext=(5,-10), fontsize=8, color="orange")

plt.xlabel("I, мА")
plt.ylabel("P, Вт / η")
plt.title("Рабочие характеристики источника")
plt.grid(True)
plt.legend()
plt.show()
