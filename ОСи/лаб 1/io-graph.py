import matplotlib.pyplot as plt
import numpy as np

# ==========================
# СТАРЫЕ ДАННЫЕ (без O3)
# ==========================

p = np.array([1, 8, 16, 32])

# Process times (sec) — без агрессивной оптимизации
proc_read_off  = np.array([3601.813, 13783, 28048, 55462]) / 1000
proc_read_on   = np.array([57471.343, 79900, 105400, 186600]) / 1000
proc_write_off = np.array([2001.165, 7800, 29200, 53200]) / 1000
proc_write_on  = np.array([66082.156, 93000, 180580, 354895]) / 1000

# Thread times (sec) — без агрессивной оптимизации
thr_read_off  = np.array([0.55, 11.24, 12.87, 15.17])
thr_read_on   = np.array([0.93, 11.18, 12.87, 14.95])
thr_write_off = np.array([1.95, 36.89, 64.95, 100.40])
thr_write_on  = np.array([1.91, 29.29, 66.36, 114.53])


# ================================
# O3 ДАННЫЕ ТОЛЬКО ДЛЯ p = 32
# ================================

p_o3 = np.array([32])

# READ / SEQ / DIRECT = OFF
proc_read_off_o3 = np.array([49.02])   # ~49.02 s
thr_read_off_o3  = np.array([0.104])   # 104.344 ms

# READ / SEQ / DIRECT = ON
proc_read_on_o3 = np.array([181.0])    # 3:01
thr_read_on_o3  = np.array([0.542])    # 542.426 ms

# WRITE / SEQ / DIRECT = OFF
proc_write_off_o3 = np.array([49.3])
thr_write_off_o3  = np.array([3.106])  # 3105.762 ms

# WRITE / SEQ / DIRECT = ON
proc_write_on_o3 = np.array([335.0])   # 5:35
thr_write_on_o3  = np.array([5.255])   # 5255.056 ms


# ==========================
# ФУНКЦИЯ ПОСТРОЕНИЯ
# ==========================

def plot_times(title, proc_time, thr_time, p_o3, proc_o3, thr_o3):
    plt.figure(figsize=(8,5))

    # старые кривые
    plt.plot(p, thr_time, marker='o', label="Thread (normal)")
    plt.plot(p, proc_time, marker='o', label="Process (normal)")

    # точки с O3 только в p=32
    plt.scatter(p_o3, thr_o3, marker='s', label="Thread O3")
    plt.scatter(p_o3, proc_o3, marker='s', label="Process O3")

    plt.xlabel("Количество потоков/процессов (p)")
    plt.ylabel("Время (секунды)")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ==========================
# 4 ГРАФИКА
# ==========================

plot_times("READ / SEQ / DIRECT = OFF",
           proc_read_off, thr_read_off,
           p_o3, proc_read_off_o3, thr_read_off_o3)

plot_times("READ / SEQ / DIRECT = ON",
           proc_read_on, thr_read_on,
           p_o3, proc_read_on_o3, thr_read_on_o3)

plot_times("WRITE / SEQ / DIRECT = OFF",
           proc_write_off, thr_write_off,
           p_o3, proc_write_off_o3, thr_write_off_o3)

plot_times("WRITE / SEQ / DIRECT = ON",
           proc_write_on, thr_write_on,
           p_o3, proc_write_on_o3, thr_write_on_o3)
