import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Параметры варианта
# -----------------------------
days = 30
new_data_per_day_mb = 300       # новые данные в сутки
compression_ratio = 3           # сжатие 3:1
retention_days = 28             # срок хранения резервных копий

# -----------------------------
# Расчёты
# -----------------------------
days_array = np.arange(1, days + 1)

# Размер базы на каждый день
db_size_mb = new_data_per_day_mb * days_array

# Размер сжатого дампа на каждый день
dump_size_mb = db_size_mb / compression_ratio

# Накопленный объём хранимых бэкапов:
# на каждый день суммируем только те дампы, которые ещё не удалены
stored_total_mb = []

for day in range(1, days + 1):
    start_day = max(1, day - retention_days + 1)
    total = dump_size_mb[start_day - 1:day].sum()
    stored_total_mb.append(total)

stored_total_mb = np.array(stored_total_mb)
stored_total_gb = stored_total_mb / 1024

# -----------------------------
# Ключевые значения для подписи
# -----------------------------
final_total_mb = stored_total_mb[-1]
final_total_gb = stored_total_gb[-1]

# -----------------------------
# Построение графиков
# -----------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Верхний график: размер ежедневного дампа
ax1.bar(days_array, dump_size_mb, label='Размер дампа', alpha=0.7)
ax1.set_title('Ежедневные резервные копии и накопленный объем')
ax1.set_ylabel('Размер дампа (МБ)')
ax1.set_xlim(0.5, days + 0.5)
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend()

# Нижний график: накопленный объём
ax2.plot(days_array, stored_total_mb, marker='o', label='Накопленный объем')
ax2.set_xlabel('Дни')
ax2.set_ylabel('Общий объем (МБ)')
ax2.set_xlim(0.5, days + 0.5)
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.legend(loc='upper left')

# Правая ось в ГБ
ax2_right = ax2.twinx()
ax2_right.set_ylabel('Общий объем (ГБ)')
ax2_right.set_ylim(ax2.get_ylim()[0] / 1024, ax2.get_ylim()[1] / 1024)

# Подпись последней точки
ax2.annotate(
    f'{final_total_gb:.1f} ГБ',
    xy=(days, final_total_mb),
    xytext=(days - 3, final_total_mb * 0.92),
    arrowprops=dict(arrowstyle='->', lw=1.5),
    fontsize=10
)

plt.tight_layout()
plt.show()

# -----------------------------
# Вывод чисел в консоль
# -----------------------------
print(f'Размер дампа на 30-й день: {dump_size_mb[-1]:.0f} МБ ({dump_size_mb[-1] / 1024:.2f} ГБ)')
print(f'Накопленный объем на 30-й день: {final_total_mb:.0f} МБ ({final_total_gb:.2f} ГБ)')