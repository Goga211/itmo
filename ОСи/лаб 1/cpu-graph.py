import matplotlib.pyplot as plt

# --- ДАННЫЕ ---

# Поточная версия (threads)
p_thr = [1, 8, 16, 32, 64]
T_thr = [32.99, 45.93, 65.90, 138.71, 273.86]
E_thr = [1.00, 5.75, 8.01, 7.61, 7.71]

# Процессная версия (processes)
p_proc = [1, 16, 32, 64]
T_proc = [29.0, 66.0, 133.0, 268.0]
E_proc = [1.00, 7.03, 6.98, 6.93]

# --- ГРАФИК 1: Время выполнения T(p) ---

plt.figure(figsize=(8, 5))
plt.plot(p_thr, T_thr, marker='o', label='Потоки (threads)')
plt.plot(p_proc, T_proc, marker='s', label='Процессы (processes)')

plt.xlabel('Число потоков / процессов p')
plt.ylabel('Время выполнения T(p), сек')
plt.title('CPU-нагрузка: время выполнения T(p)')
plt.xticks([1, 8, 16, 32, 64])
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()

# --- ГРАФИК 2: Эффективность E(p) ---

plt.figure(figsize=(8, 5))
plt.plot(p_thr, E_thr, marker='o', label='Потоки (threads)')
plt.plot(p_proc, E_proc, marker='s', label='Процессы (processes)')

plt.xlabel('Число потоков / процессов p')
plt.ylabel('Эффективность E(p)')
plt.title('CPU-нагрузка: эффективность E(p)')
plt.xticks([1, 8, 16, 32, 64])
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()

plt.show()
