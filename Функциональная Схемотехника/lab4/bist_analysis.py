#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Анализ распределения операндов в режиме BIST
Лабораторная работа №4 (Альтернативная визуализация)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# =============================================================================
# Настройки отображения
# =============================================================================
rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False
plt.style.use("ggplot")

# =============================================================================
# LFSR 8 бит
# =============================================================================
class LFSR8:
    def __init__(self, poly, seed):
        self.poly = poly
        self.state = seed
        self.history = [seed]

    def next(self):
        feedback = self.state >> 7
        for i in range(7):
            if (self.poly >> i) & 1:
                feedback ^= (self.state >> i) & 1
        self.state = ((self.state << 1) | feedback) & 0xFF
        self.history.append(self.state)
        return self.state

# =============================================================================
# Кубический корень
# =============================================================================
def cube_root(x):
    if x == 0:
        return 0
    y = 1
    for _ in range(20):
        y_new = (2*y + x//(y*y)) // 3
        if y_new >= y:
            break
        y = y_new
    return y

# =============================================================================
# Проверка допустимости
# =============================================================================
def check_valid_operands(a, b):
    if b == 0:
        return False, "b = 0"
    cbrt_b = cube_root(b)
    result = a * cbrt_b
    if result >= 65536:
        return False, "overflow"
    return True, "valid"

# =============================================================================
# Параметры BIST
# =============================================================================
LFSR1_POLY = 0b01011000
LFSR2_POLY = 0b11001000

LFSR1_SEED = 0xA5
LFSR2_SEED = 0x5A

ITERATIONS = 256

# =============================================================================
# Генерация LFSR
# =============================================================================
print("=" * 60)
print("  Анализ распределения операндов BIST")
print("=" * 60)

lfsr1 = LFSR8(LFSR1_POLY, LFSR1_SEED)
lfsr2 = LFSR8(LFSR2_POLY, LFSR2_SEED)

a_values = [LFSR1_SEED]
b_values = [LFSR2_SEED]

for _ in range(ITERATIONS - 1):
    a_values.append(lfsr1.next())
    b_values.append(lfsr2.next())

print(f"Сгенерировано пар: {len(a_values)}")

# =============================================================================
# Анализ допустимости
# =============================================================================
valid_count = 0
invalid_reasons = {}

for a, b in zip(a_values, b_values):
    ok, reason = check_valid_operands(a, b)
    if ok:
        valid_count += 1
    else:
        invalid_reasons[reason] = invalid_reasons.get(reason, 0) + 1

valid_percentage = valid_count / len(a_values) * 100

print(f"\nДопустимых: {valid_count} / {len(a_values)}")
print(f"Процент: {valid_percentage:.2f}%")

# =============================================================================
# Статистика
# =============================================================================
print("\nСтатистика a:")
print(f"Мин: {min(a_values)} Макс: {max(a_values)}")
print(f"Среднее: {np.mean(a_values):.2f}")
print(f"Медиана: {np.median(a_values):.2f}")

print("\nСтатистика b:")
print(f"Мин: {min(b_values)} Макс: {max(b_values)}")
print(f"Среднее: {np.mean(b_values):.2f}")
print(f"Медиана: {np.median(b_values):.2f}")

# =============================================================================
# Подготовка данных для графиков
# =============================================================================
valid_a = [a for a, b in zip(a_values, b_values) if check_valid_operands(a, b)[0]]
valid_b = [b for a, b in zip(a_values, b_values) if check_valid_operands(a, b)[0]]
invalid_a = [a for a, b in zip(a_values, b_values) if not check_valid_operands(a, b)[0]]
invalid_b = [b for a, b in zip(a_values, b_values) if not check_valid_operands(a, b)[0]]

# =============================================================================
# Альтернативные графики
# =============================================================================
fig, axs = plt.subplots(2, 2, figsize=(15, 11))
fig.suptitle("Альтернативная визуализация BIST", fontsize=16, fontweight="bold")

# STEP-гистограмма a
axs[0, 0].hist(a_values, bins=32, histtype="step", linewidth=2)
axs[0, 0].axvline(np.median(a_values), linestyle=":", linewidth=2, label="Медиана")
axs[0, 0].set_title("STEP-гистограмма a")
axs[0, 0].set_xlabel("a")
axs[0, 0].set_ylabel("Частота")
axs[0, 0].legend()

# STEP-гистограмма b
axs[0, 1].hist(b_values, bins=32, histtype="stepfilled", alpha=0.4)
axs[0, 1].axvline(np.median(b_values), linestyle=":", linewidth=2, label="Медиана")
axs[0, 1].set_title("STEP-гистограмма b")
axs[0, 1].set_xlabel("b")
axs[0, 1].set_ylabel("Частота")
axs[0, 1].legend()

# HEXBIN плотность
hb = axs[1, 0].hexbin(a_values, b_values, gridsize=30)
axs[1, 0].set_title("HEXBIN плотность (a, b)")
axs[1, 0].set_xlabel("a")
axs[1, 0].set_ylabel("b")
cb = fig.colorbar(hb, ax=axs[1, 0])
cb.set_label("Количество")

# STEM покрытие
x = np.arange(256)
a_cov = np.array([1 if i in a_values else 0 for i in x])
b_cov = np.array([1 if i in b_values else 0 for i in x])

axs[1, 1].stem(x, a_cov, linefmt="--", markerfmt="o", basefmt=" ")
axs[1, 1].stem(x, b_cov, linefmt=":", markerfmt="x", basefmt=" ")
axs[1, 1].set_title("Покрытие диапазона")
axs[1, 1].set_xlabel("Значение")
axs[1, 1].set_ylabel("Покрыт (1/0)")
axs[1, 1].set_xlim(0, 255)

plt.tight_layout()
plt.savefig("bist_distribution_alt.png", dpi=300, bbox_inches="tight")

# =============================================================================
# Альтернативная карта допустимости
# =============================================================================
fig2, ax = plt.subplots(figsize=(10, 9))

validity_map = np.zeros((256, 256))
for a in range(256):
    for b in range(256):
        validity_map[b, a] = 1 if check_valid_operands(a, b)[0] else 0

mesh = ax.pcolormesh(validity_map, shading="nearest")
ax.contour(validity_map, levels=[0.5], linewidths=2)

ax.set_title("Контурная карта допустимых значений")
ax.set_xlabel("a")
ax.set_ylabel("b")

plt.colorbar(mesh, ax=ax, label="Допустимость")

ax.scatter(valid_a, valid_b, s=12, alpha=0.8, label="Допустимые")
if invalid_a:
    ax.scatter(invalid_a, invalid_b, s=25, marker="x", label="Недопустимые")

ax.legend()
plt.tight_layout()
plt.savefig("bist_validity_map_alt.png", dpi=300, bbox_inches="tight")

plt.show()

print("\n" + "=" * 60)
print("  Анализ завершён!")
print("=" * 60)
