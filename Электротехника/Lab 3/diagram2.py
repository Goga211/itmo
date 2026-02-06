import numpy as np
import matplotlib.pyplot as plt

# --- Исходные данные ---
R1 = 48
Rk = 15
L = 51.515e-3
C = 21.881e-6
f = 1267.621  # Гц

# --- Расчёт реактивных сопротивлений ---
XL = 2 * np.pi * f * L
XC = 1 / (2 * np.pi * f * C)

# --- Углы токов (в градусах) ---
psi_I1 = -np.degrees(np.arctan(XL / Rk))
psi_I2 =  np.degrees(np.arctan(XC / R1))

print(f"XL = {XL:.3f} Ом,  XC = {XC:.3f} Ом")
print(f"ψ_I1 = {psi_I1:.2f}°, ψ_I2 = {psi_I2:.2f}°")

# --- Амплитуды из таблицы ---
I  = 0.206
I1 = 0.207
I2 = 0.024

# --- Векторное представление токов ---
I1_vec = I1 * np.exp(1j * np.radians(psi_I1))
I2_vec = I2 * np.exp(1j * np.radians(psi_I2))
I_vec  = I1_vec + I2_vec

# --- Визуализация ---
fig, ax = plt.subplots(figsize=(6,6))

def draw_vec(start, vec, color, label):
    ax.arrow(start.real, start.imag, vec.real, vec.imag,
             head_width=0.02, head_length=0.04,
             fc=color, ec=color, length_includes_head=True)
    ax.text(start.real + vec.real*0.6,
            start.imag + vec.imag*0.6,
            label, color=color, fontsize=12, fontweight='bold')

# --- Построение ---
draw_vec(0+0j, I_vec, 'red', 'I')
draw_vec(0+0j, I1_vec, 'blue', 'I1')
draw_vec(0+0j, I2_vec, 'orange', 'I2')

# --- Настройки осей ---
ax.axhline(0, color='black', linewidth=1)
ax.axvline(0, color='black', linewidth=1)
ax.set_aspect('equal', 'box')
ax.grid(True)
ax.set_xlabel("Re")
ax.set_ylabel("Im")
ax.set_title("Векторная диаграмма токов, f = 1267.621 Гц")

# --- Масштаб ---
max_r = max(abs(I_vec), abs(I1_vec), abs(I2_vec)) * 1.3
ax.set_xlim(-max_r, max_r)
ax.set_ylim(-max_r, max_r)

plt.show()
