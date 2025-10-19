# -*- coding: utf-8 -*-
"""
УИР №1 — полный перерасчёт по своей выборке.
Что считает скрипт:
1) Таблица "Форма 1" для N ∈ {10,20,50,100,200,300}: m, s^2, s, CV, ДИ 0.90/0.95/0.99
2) АК(лаг=1..10) для исходной последовательности + график
3) Гистограмма (18 равных интервалов) + таблица интервалов и частот + график
4) Аппроксимация законом Эрланга по двум моментам (k, λ) + наложение pdf на гистограмму
5) Генерация 300 значений по Эрлангу(k, λ), расчёт таблицы как в (1), АК, корреляция между исходной и сгенерированной
Все таблицы сохраняются в CSV, графики — в PNG.
"""

import math
import random
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============ ВСТАВЬ СВОЮ ВЫБОРКУ НИЖЕ (как есть, с запятыми) ============
RAW_DATA = r"""
277,64
153,44
88,76
87,32
94,68
141,38
295,00
257,69
251,29
20,03
156,60
63,98
303,63
86,80
208,94
283,83
221,60
86,55
90,44
352,12
16,71
169,27
47,46
275,90
92,27
134,89
132,00
329,63
192,69
223,41
202,01
110,37
209,36
180,89
288,69
100,43
150,41
199,82
116,65
257,17
132,01
14,84
51,80
417,11
7,93
62,50
174,34
138,25
159,72
96,41
116,75
96,13
164,83
68,31
420,38
118,45
132,66
222,87
141,33
115,42
58,95
38,14
172,52
203,79
116,39
74,40
22,55
92,37
118,79
126,88
39,68
184,73
170,34
133,07
292,58
35,07
35,91
364,56
99,32
170,04
439,06
322,48
490,40
364,58
58,49
72,01
221,79
183,21
22,33
98,12
140,25
470,49
357,77
544,95
73,93
152,33
52,64
138,15
38,86
127,92
49,15
323,16
195,03
129,11
183,68
441,19
153,09
199,64
79,14
138,73
83,84
89,33
149,78
214,79
213,52
223,12
190,14
40,16
189,74
243,52
112,04
340,32
403,79
584,31
380,72
173,84
236,19
239,46
160,43
267,02
155,02
11,62
241,30
22,61
58,05
39,05
243,64
458,96
100,75
327,91
136,96
371,46
219,16
329,67
115,96
168,63
296,86
119,74
149,14
192,51
176,36
117,96
190,67
62,13
171,45
330,87
405,41
223,72
239,71
199,50
49,79
61,67
112,07
72,04
99,70
292,19
150,96
198,00
201,33
160,27
80,21
189,50
144,68
175,17
129,94
142,44
107,20
86,71
56,35
43,21
140,62
304,19
178,22
137,92
51,30
106,31
153,14
236,05
220,22
376,03
89,41
97,39
161,00
146,66
124,59
414,91
204,84
142,83
267,16
48,35
348,50
268,90
191,54
209,63
486,86
229,65
153,85
168,69
109,11
288,02
138,90
255,62
227,56
97,95
88,85
255,19
152,84
100,12
140,07
201,98
119,39
175,92
90,09
182,25
146,42
204,48
293,44
219,74
380,66
195,67
266,86
199,47
101,43
365,20
270,03
177,41
175,62
324,27
289,65
139,26
212,78
132,67
256,16
181,94
119,39
90,81
94,16
301,45
101,86
237,38
52,35
13,09
270,56
637,91
99,35
97,00
214,06
215,16
166,62
144,14
173,57
143,08
478,02
323,03
284,34
46,33
141,57
202,88
110,09
49,68
147,41
334,28
146,81
91,83
315,39
170,02
79,54
144,28
248,54
361,40
43,65
650,38
116,38
93,09
436,24
209,20
192,72
214,61
70,25
138,28
42,58
321,33
151,36
125,37
266,43
364,76
61,75
10,13
193,05
10,127
""".strip()

# ========================= УТИЛИТЫ =========================

def parse_numbers(raw: str) -> np.ndarray:
    """Парсинг чисел с десятичной запятой, разделённых переводами строк/пробелами/запятыми."""
    # заменим запятую на точку и разобьём по любому пробельному/знаку
    # простая стратегия: заменить запятые в десятичной записи на точку и splitlines(),
    # затем удалить пустые, запустить float() для каждого.
    # Также поддержим случайные запятые-разделители: заменим '\n' на пробел, потом split()
    cleaned = raw.replace(',', '.').replace('\r', '\n')
    tokens = []
    for line in cleaned.splitlines():
        line = line.strip()
        if not line:
            continue
        # Если строка содержит несколько чисел через пробел — разобьём
        for tok in line.split():
            tokens.append(tok)
    nums = []
    for t in tokens:
        t = t.strip()
        if not t:
            continue
        # защита от хвостовых запятых/точек
        if t.endswith(',') or t.endswith('.'):
            t = t[:-1]
        try:
            nums.append(float(t))
        except ValueError:
            # пропускаем мусорные токены, если вдруг встретятся
            pass
    return np.array(nums, dtype=float)


def descriptive_stats(x: np.ndarray) -> Dict[str, float]:
    """Возвращает m, var (unbiased), std, cv для массива x."""
    n = len(x)
    m = float(np.mean(x))
    var = float(np.var(x, ddof=1)) if n > 1 else float('nan')
    std = math.sqrt(var) if var == var else float('nan')  # проверка на nan
    cv = std / m if (m != 0 and std == std) else float('nan')
    return {"n": n, "mean": m, "var": var, "std": std, "cv": cv}


# Предзаданные z-квантили нормального распределения
Z_CRIT = {
    0.90: 1.6448536269514722,
    0.95: 1.959963984540054,
    0.99: 2.5758293035489004,
}

# При желании можно использовать t-квантили для ровно нужных n (df=n-1).
# Ниже — t-квантили для df = 9,19,49,99,199,299.
# Источник значений — стандартные таблицы Стьюдента (обобщённо; округлены до 5–6 знаков).
T_CRIT = {
    9:  {0.90: 1.833, 0.95: 2.262, 0.99: 3.250},
    19: {0.90: 1.729, 0.95: 2.093, 0.99: 2.861},
    49: {0.90: 1.676, 0.95: 2.009, 0.99: 2.678},
    99: {0.90: 1.660, 0.95: 1.984, 0.99: 2.626},
    199:{0.90: 1.653, 0.95: 1.972, 0.99: 2.601},
    299:{0.90: 1.650, 0.95: 1.968, 0.99: 2.589},
}

def confint_mean(x: np.ndarray, levels=(0.90,0.95,0.99), use_t=False) -> Dict[float, float]:
    """Возвращает полу-ширины доверительных интервалов для среднего на заданных уровнях.
       Если use_t=True и df есть в T_CRIT, берётся t-квантиль, иначе z-квантиль."""
    n = len(x)
    stats = descriptive_stats(x)
    se = stats["std"] / math.sqrt(n) if n > 1 else float('nan')
    df = n - 1
    out = {}
    for p in levels:
        if use_t and df in T_CRIT and p in T_CRIT[df]:
            q = T_CRIT[df][p]
        else:
            q = Z_CRIT[p]
        out[p] = q * se
    return out


def acf(x: np.ndarray, max_lag: int = 10) -> List[float]:
    """Несмещённая автокорреляция по лагам 1..max_lag."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    m = np.mean(x)
    denom = np.sum((x - m)**2)
    r = []
    for k in range(1, max_lag+1):
        if k >= n:
            r.append(float('nan'))
            continue
        num = np.sum((x[k:] - m) * (x[:-k] - m))
        r.append(float(num / denom) if denom != 0 else float('nan'))
    return r


def histogram_table(x: np.ndarray, bins: int = 18) -> pd.DataFrame:
    """Равные интервалы от min до max и частоты."""
    xmin, xmax = float(np.min(x)), float(np.max(x))
    edges = np.linspace(xmin, xmax, bins + 1)
    counts, _ = np.histogram(x, bins=edges)
    df = pd.DataFrame({
        "№": np.arange(1, bins+1),
        "Левая граница": edges[:-1],
        "Правая граница": edges[1:],
        "Частота": counts
    })
    return df


def erlang_params_by_moments(x: np.ndarray, k_rule: str = "round") -> Tuple[int, float, float]:
    """
    Оценка параметров Эрланга по двум моментам.
    k ≈ (m/s)^2, λ = k/m, θ = 1/λ. k_rule: "round"|"floor"|"ceil".
    Возвращает (k, λ, θ).
    """
    st = descriptive_stats(x)
    m, s = st["mean"], st["std"]
    if not (m > 0 and s > 0):
        raise ValueError("mean/std not positive; cannot fit Erlang by moments")
    k_hat = (m / s) ** 2  # = 1/CV^2
    if k_rule == "floor":
        k = max(1, int(math.floor(k_hat)))
    elif k_rule == "ceil":
        k = max(1, int(math.ceil(k_hat)))
    else:
        k = max(1, int(round(k_hat)))
    lam = k / m
    theta = 1.0 / lam
    return k, lam, theta


def erlang_pdf_grid(x_grid: np.ndarray, k: int, lam: float) -> np.ndarray:
    """Плотность Эрланга на сетке x_grid (k — целое >=1, lam>0)."""
    # f(x) = λ^k x^{k-1} e^{-λ x} / (k-1)!  для x>=0
    x = np.maximum(x_grid, 0.0)
    from math import exp
    fact = math.factorial(k-1)
    lamk = lam**k
    # чтобы уменьшить переполнение — считаем через логарифмы при необходимости;
    # но для наших диапазонов обычно достаточно прямого варианта:
    f = lamk * (x ** (k-1)) * np.exp(-lam * x) / fact
    return f


def generate_erlang(n: int, k: int, lam: float, seed: int = 42) -> np.ndarray:
    """Генерация n значений Эрланга (как суммы k экспонент) через numpy.random.gamma."""
    rng = np.random.default_rng(seed)
    theta = 1.0 / lam
    # numpy параметризует гамма(shape=k, scale=θ). Для целого k это Эквивалент Эрланга.
    return rng.gamma(shape=float(k), scale=float(theta), size=n)


def table_form1_for_prefixes(x: np.ndarray,
                             Ns=(10, 20, 50, 100, 200, 300),
                             use_t=False) -> pd.DataFrame:
    """Строит таблицу 'Форма 1' для префиксных подвыборок размеров Ns."""
    rows = []
    for N in Ns:
        if N > len(x):
            continue
        xN = x[:N]
        st = descriptive_stats(xN)
        cis = confint_mean(xN, use_t=use_t)
        rows.append({
            "N": N,
            "m": st["mean"],
            "s2": st["var"],
            "s": st["std"],
            "CV": st["cv"],
            "CI90(±)": cis[0.90],
            "CI95(±)": cis[0.95],
            "CI99(±)": cis[0.99],
        })
    df = pd.DataFrame(rows)
    return df


def save_df(df: pd.DataFrame, path: str):
    df_rounded = df.copy()
    for c in df_rounded.columns:
        if df_rounded[c].dtype.kind in "fc":
            df_rounded[c] = df_rounded[c].astype(float).round(5)
    df_rounded.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[+] Saved: {path}")


def main():
    data = parse_numbers(RAW_DATA)
    if len(data) == 0:
        raise SystemExit("Пустые данные — проверь RAW_DATA")

    print(f"Размер выборки: n={len(data)}")
    # ---------- Этап 1: Форма 1 ----------
    form1 = table_form1_for_prefixes(data, Ns=(10,20,50,100,200,300), use_t=True)
    save_df(form1, "form1_prefix_stats.csv")
    print(form1)

    # ---------- Этап 3: АК (лаг 1..10) ----------
    r = acf(data, max_lag=10)
    acf_df = pd.DataFrame({"лаг": list(range(1, 11)), "АК": r})
    save_df(acf_df, "acf_lags_1_10.csv")
    # График АК
    plt.figure(figsize=(8,4))
    plt.stem(acf_df["лаг"].values, acf_df["АК"].values, basefmt=" ")
    plt.title("АК исходной последовательности (лаг 1..10)")
    plt.xlabel("Лаг")
    plt.ylabel("АК")
    plt.tight_layout()
    plt.savefig("acf_original.png", dpi=200)
    plt.close()
    print("[+] Saved: acf_original.png")

    # ---------- Этап 4: Гистограмма (18 равных интервалов) ----------
    hist_df = histogram_table(data, bins=18)
    save_df(hist_df, "histogram_18bins.csv")
    # График гистограммы
    plt.figure(figsize=(8,4))
    edges = np.linspace(float(np.min(data)), float(np.max(data)), 18+1)
    plt.hist(data, bins=edges)
    plt.title("Гистограмма исходной последовательности (18 интервалов)")
    plt.xlabel("Значение")
    plt.ylabel("Частота")
    plt.tight_layout()
    plt.savefig("hist_original.png", dpi=200)
    plt.close()
    print("[+] Saved: hist_original.png")

    # ---------- Этап 5: Эрланг по двум моментам ----------
    k, lam, theta = erlang_params_by_moments(data, k_rule="round")
    print(f"Оценка Эрланга по двум моментам: k={k}, λ={lam:.8f}, θ={theta:.5f}")
    # Наложение pdf на гистограмму
    plt.figure(figsize=(8,4))
    plt.hist(data, bins=edges, density=True)
    grid = np.linspace(0, float(np.max(data)), 1000)
    pdf = erlang_pdf_grid(grid, k=k, lam=lam)
    plt.plot(grid, pdf)
    plt.title(f"Наложение pdf Эрланга(k={k}, λ={lam:.5f}) на гистограмму")
    plt.xlabel("x")
    plt.ylabel("density / freq")
    plt.tight_layout()
    plt.savefig("hist_with_erlang_pdf.png", dpi=200)
    plt.close()
    print("[+] Saved: hist_with_erlang_pdf.png")

    # ---------- Этап 6–7: Генерация по Эрлангу и сравнение ----------
    sim = generate_erlang(n=300, k=k, lam=lam, seed=42)
    # Таблица как Форма 1 по префиксам
    form1_sim = table_form1_for_prefixes(sim, Ns=(10,20,50,100,200,300), use_t=True)
    save_df(form1_sim, "form1_prefix_stats_simulated.csv")
    # АК сгенерированной
    r_sim = acf(sim, max_lag=10)
    acf_sim_df = pd.DataFrame({"лаг": list(range(1, 11)), "АК": r_sim})
    save_df(acf_sim_df, "acf_sim_lags_1_10.csv")
    # График АК сгенерированной
    plt.figure(figsize=(8,4))
    plt.stem(acf_sim_df["лаг"].values, acf_sim_df["АК"].values, basefmt=" ")
    plt.title("АК сгенерированной Эрланг-последовательности (лаг 1..10)")
    plt.xlabel("Лаг")
    plt.ylabel("АК")
    plt.tight_layout()
    plt.savefig("acf_simulated.png", dpi=200)
    plt.close()
    print("[+] Saved: acf_simulated.png")

    # Корреляция между исходной и сгенерированной (по первым 300 значениям)
    n_cmp = min(len(data), len(sim))
    corr = float(np.corrcoef(data[:n_cmp], sim[:n_cmp])[0,1])
    with open("corr_original_vs_sim.txt", "w", encoding="utf-8") as f:
        f.write(f"Коэффициент корреляции (исходная vs сгенерированная): {corr:.6f}\n")
    print(f"[+] Saved: corr_original_vs_sim.txt  (r={corr:.6f})")

    # Гистограмма сгенерированной
    plt.figure(figsize=(8,4))
    edges_sim = np.linspace(float(np.min(sim)), float(np.max(sim)), 18+1)
    plt.hist(sim, bins=edges_sim)
    plt.title("Гистограмма сгенерированной Эрланг-последовательности (18 интервалов)")
    plt.xlabel("Значение")
    plt.ylabel("Частота")
    plt.tight_layout()
    plt.savefig("hist_simulated.png", dpi=200)
    plt.close()
    print("[+] Saved: hist_simulated.png")

    # Сохраним сводку параметров в CSV
    summary = pd.DataFrame([
        {
            "k": k,
            "lambda": lam,
            "theta": theta,
            "mean_original": descriptive_stats(data)["mean"],
            "std_original": descriptive_stats(data)["std"],
            "cv_original": descriptive_stats(data)["cv"],
            "mean_sim": descriptive_stats(sim)["mean"],
            "std_sim": descriptive_stats(sim)["std"],
            "cv_sim": descriptive_stats(sim)["cv"],
            "corr_original_vs_sim": corr,
        }
    ])
    save_df(summary, "summary_erlang_fit_and_compare.csv")

    print("\nГотово. Файлы сохранены в текущей папке:")
    print(" - form1_prefix_stats.csv, form1_prefix_stats_simulated.csv")
    print(" - acf_lags_1_10.csv, acf_sim_lags_1_10.csv, acf_original.png, acf_simulated.png")
    print(" - histogram_18bins.csv, hist_original.png, hist_simulated.png")
    print(" - hist_with_erlang_pdf.png")
    print(" - corr_original_vs_sim.txt")
    print(" - summary_erlang_fit_and_compare.csv")

if __name__ == "__main__":
    # Для воспроизводимости некоторых операций:
    random.seed(42)
    np.random.seed(42)
    main()
