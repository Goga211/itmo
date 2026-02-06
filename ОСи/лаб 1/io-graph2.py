import matplotlib.pyplot as plt
import numpy as np

p = np.array([1, 8, 16, 32])

# PROCESS EFFICIENCIES
E_proc_read_off  = np.array([1.00, 2.09, 2.05, 2.07])
E_proc_read_on   = np.array([1.00, 5.75, 8.73, 9.86])
E_proc_write_off = np.array([1.00, 2.05, 1.10, 1.20])
E_proc_write_on  = np.array([1.00, 5.68, 5.85, 5.96])

# THREAD EFFICIENCIES — YOUR EXACT DATA
E_thr_read_off  = np.array([1.00, 0.39, 0.68, 1.16])
E_thr_read_on   = np.array([1.00, 0.67, 1.16, 1.99])
E_thr_write_off = np.array([1.00, 0.42, 0.48, 0.62])
E_thr_write_on  = np.array([1.00, 0.52, 0.46, 0.53])

def plot_eff(title, E_proc, E_thr):
    plt.figure(figsize=(8,5))
    plt.plot(p, E_proc, marker='o', label="Process efficiency E(p)")
    plt.plot(p, E_thr, marker='o', label="Thread efficiency E(p)")
    plt.xlabel("p (threads/processes)")
    plt.ylabel("Efficiency E(p)")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_eff("READ / SEQ / DIRECT = OFF — Efficiency", E_proc_read_off, E_thr_read_off)
plot_eff("READ / SEQ / DIRECT = ON — Efficiency",  E_proc_read_on,  E_thr_read_on)
plot_eff("WRITE / SEQ / DIRECT = OFF — Efficiency", E_proc_write_off, E_thr_write_off)
plot_eff("WRITE / SEQ / DIRECT = ON — Efficiency",  E_proc_write_on,  E_thr_write_on)
