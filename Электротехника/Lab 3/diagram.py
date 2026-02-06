import numpy as np
import matplotlib.pyplot as plt

# === ПАРАМЕТРЫ ТВОЕГО ВАРИАНТА ===
U_amp = 10.0          # В
R1    = 48.0          # Ом
Rk    = 15.0          # Ом (сопротивление катушки)
L     = 51.515e-3     # Гн
C     = 21.881e-6     # Ф

def calc_phasors(f_hz: float):
    """
    Считаем комплексные напряжения на R1, катушке и конденсаторе
    для заданной частоты f_hz.
    """
    w = 2 * np.pi * f_hz

    Z_R1 = R1                         # просто действительное
    Z_k  = Rk + 1j * w * L            # Rk + jωL
    Z_C  = -1j / (w * C)              # -j / (ωC)

    Z_total = Z_R1 + Z_k + Z_C        # последовательно
    I = U_amp / Z_total               # комплексный ток

    U_R1 = I * Z_R1
    U_k  = I * Z_k
    U_C  = I * Z_C
    U    = I * Z_total                # должно ≈ 10∠0

    return U, U_R1, U_k, U_C


def plot_vector_diagram(f_hz: float):
    """
    Строим векторную диаграмму для заданной частоты f_hz.
    """
    U, U_R1, U_k, U_C = calc_phasors(f_hz)

    # Точки многоугольника: 0 -> U_C -> U_C+U_R1 -> U_C+U_R1+U_k (= U)
    z0 = 0 + 0j
    z1 = U_C
    z2 = U_C + U_R1
    z3 = U_C + U_R1 + U_k   # должно совпасть с U

    fig, ax = plt.subplots(figsize=(6, 6))

    def draw_vec(start, end, color, label, tpos=0.55):
        dx = end.real - start.real
        dy = end.imag - start.imag
        ax.arrow(start.real, start.imag, dx, dy,
                 length_includes_head=True,
                 head_width=0.25, head_length=0.5,
                 ec=color, fc=color)
        # подпись примерно посередине вектора
        x_text = start.real + dx * tpos
        y_text = start.imag + dy * tpos
        ax.text(x_text, y_text, label)

    # U (от 0 до z3)
    draw_vec(z0, U,   'red',   'U')
    # Uc (от 0 до z1)
    draw_vec(z0, z1,  'blue',  'U_C')
    # UR1 (от z1 до z2)
    draw_vec(z1, z2,  'orange','U_R1')
    # Uk (от z2 до z3)
    draw_vec(z2, z3,  'green', 'U_k')

    # Оси
    ax.axhline(0, color='black', linewidth=1)
    ax.axvline(0, color='black', linewidth=1)

    # Красиво поджать масштаб
    xs = [z0.real, z1.real, z2.real, z3.real, U.real]
    ys = [z0.imag, z1.imag, z2.imag, z3.imag, U.imag]
    r_max = max(max(map(abs, xs)), max(map(abs, ys))) + 2

    ax.set_xlim(-r_max, r_max)
    ax.set_ylim(-r_max, r_max)
    ax.set_aspect('equal', 'box')
    ax.grid(True)

    ax.set_xlabel('Re')
    ax.set_ylabel('Im')
    ax.set_title(f'Векторная диаграмма напряжений, f = {f_hz:.3f} Гц')

    plt.show()


# === ПРИМЕР ИСПОЛЬЗОВАНИЯ ===
# Диаграмма для резонансной частоты (строка f0 = 149.906 из твоей таблицы)
plot_vector_diagram(149.906)

# Можно попробовать любую частоту из таблицы, например:
# plot_vector_diagram(59.962)
# plot_vector_diagram(164.897)
# plot_vector_diagram(299.812)
