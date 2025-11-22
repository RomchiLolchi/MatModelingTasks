# Задание 6 - решение нелинейных уравнений
# todo НОРМАЛИЗАЦИЯ ЕЩЁ РАЗ!!!! (проверка с большим/меньшим усечением; с параметрами)
import math
from datetime import datetime
from numpy import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.special import erf

N_default = 1000
N_input = input(f"Введите N (объём выборки; по умолчанию {N_default}): ")
N = int(N_input) if N_input else N_default

random.seed(int(datetime.now().timestamp()))

a_default, b_default = -10.0, 10.0
a = float(input(f"Введите a (по умолчанию {a_default}): ") or a_default)
b = float(input(f"Введите b (по умолчанию {b_default}): ") or b_default)

r_base = random.random(size=N)
x_uniform = a + (b - a) * r_base

uniform_mean = (a + b) / 2
uniform_var = (b - a) ** 2 / 12

m_default, D2_default, n_default = 0.0, 1.0, 12
m = float(input(f"Введите m (по умолчанию {m_default}): ") or m_default)
D2 = float(input(f"Введите D2 (по умолчанию {D2_default}): ") or D2_default)
n = int(input(f"Введите n (по умолчанию {n_default}): ") or n_default)

mv = n / 2
Dv = math.sqrt(n / 12)

x_gauss = np.array([(sum(random.random(size=n)) - mv) / Dv * math.sqrt(D2) + m for _ in range(N)])

bcoef_default = 3.0
bcoef = float(input(f"Введите параметр b (по умолчанию {bcoef_default}): ") or bcoef_default)

mrter = math.sqrt(math.pi * bcoef**2 / 2)
disprter =  (2 - math.pi/2) * bcoef**2


def rayleigh_pdf_vector(x, b):
    return np.where(x >= 0, (x / (b * b)) * np.exp(-(x**2) / (2 * b**2)), 0)


def rayleigh_pdf(x, b):
    return (x / (b * b)) * np.exp(-(x * x) / (2 * b * b)) if x >= 0 else 0


if a <= bcoef <= b:
    M = rayleigh_pdf(bcoef, bcoef)
else:
    M = max(rayleigh_pdf(a, bcoef), rayleigh_pdf(b, bcoef))

x_rayleigh = []
while len(x_rayleigh) < N:
    X = a + (b - a) * random.random()
    Y = M * random.random()
    if Y <= rayleigh_pdf(X, bcoef):
        x_rayleigh.append(X)
x_rayleigh = np.array(x_rayleigh)


def get_moments(arr, m_theor):
    N = len(arr)

    S1 = 0
    for x in arr:
        S1 += x
    m_sample = S1 / N

    S2 = 0
    for x in arr:
        S2 += (x - m_sample) ** 2
    D_sample = S2 / N

    D_theor = 0
    for x in arr:
        D_theor += (x - m_theor) ** 2
    D_theor = D_theor / N

    return m_sample, D_sample, D_theor


mom_u, var_u, d_theor_u = get_moments(x_uniform, uniform_mean)
mom_g, var_g, d_theor_g = get_moments(x_gauss, m)
mom_r, var_r, d_theor_r = get_moments(x_rayleigh, mrter)


def manual_hist(data, title, pdf_func=None, cdf_func=None):
    data = np.sort(data)

    k = max(9, min(21, int(4 * math.log10(N))))
    xmin, xmax = data[0], data[-1]
    A = xmin * 0.98
    B = xmax * 1.02
    dx = (B - A) / k

    intervals = [(A + i * dx, A + (i + 1) * dx) for i in range(k)]
    freq = [sum((data >= low) & (data < high)) for low, high in intervals]
    delta = [f / (N * dx) for f in freq]

    centers = [(low + high) / 2 for low, high in intervals]

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.bar(centers, delta, width=dx, edgecolor="black", alpha=0.6)
    plt.title(f"Гистограмма: {title}")
    plt.xlabel("x")
    plt.ylabel("Относительная частота")

    if pdf_func:
        x_plot = np.linspace(A, B, N)
        plt.plot(x_plot, pdf_func(x_plot), linewidth=2)

    cumsum = np.cumsum(freq) / N
    plt.subplot(1, 2, 2)
    x_vals = [low for low, _ in intervals] + [intervals[-1][1]]
    y_vals = [0] + list(cumsum)

    plt.step(x_vals, y_vals)

    if cdf_func:
        x_plot = np.linspace(A, B, N)
        plt.plot(x_plot, cdf_func(x_plot), linewidth=2)

    plt.title(f"Полигон накопленных частот: {title}")
    plt.xlabel("x")
    plt.ylabel("F(x)")
    plt.tight_layout()
    plt.show()


manual_hist(
    x_uniform,
    "Равномерное распределение",
    pdf_func=lambda x: np.where((x >= a) & (x <= b), 1 / (b - a), 0),
    cdf_func=lambda x: np.clip((x - a) / (b - a), 0, 1)
)

manual_hist(
    x_gauss,
    "Нормальное распределение (ЦПТ)",
    pdf_func=lambda x: 1 / np.sqrt(2 * np.pi * D2) * np.exp(-(x - m) ** 2 / (2 * D2)),
    cdf_func=lambda x: 0.5 * (1 + erf((x - m) / np.sqrt(2 * D2)))
)

Z, _ = quad(lambda x: rayleigh_pdf(x, bcoef), a, b)
manual_hist(
    x_rayleigh,
    "Релеевское распределение",
    # УРЕЗКА + НОРМИРОВКА
    # pdf_func=lambda x: np.where((x >= a) & (x <= b), rayleigh_pdf_vector(x, bcoef) / Z, 0),
    # cdf_func=lambda x: np.where(x < a, 0, np.where(x > b, 1, np.array(
    #     [quad(lambda t: rayleigh_pdf_vector(t, bcoef), a, xi)[0] / Z for xi in x])))
    # ОБЫЧНЫЕ ФУНКЦИИ:
    # pdf_func=lambda x: np.where(x >= 0, (x / (bcoef ** 2)) * np.exp(-x**2 / (2 * bcoef**2)), 0),
    # cdf_func=lambda x: np.where(x >= 0, 1 - np.exp(-x**2 / (2 * bcoef**2)), 0),
    # НОРМИРОВКА
    pdf_func=lambda x: np.where(x >= 0, rayleigh_pdf_vector(x, bcoef) / Z, 0),
    cdf_func=lambda x: np.where(x >= 0, np.array(
         [quad(lambda t: rayleigh_pdf_vector(t, bcoef), a, xi)[0] / Z for xi in x]), 0)
)

print("\n===== РЕЗУЛЬТАТЫ =====")
print(
    f"Равномерное:   M выборочное={mom_u:.4f}, D выборочная={var_u:.7f}, D с известным m={d_theor_u:.7f}, M теор={uniform_mean:.4f}, D теор={uniform_var:.4f}")
print(
    f"Гауссовское:   M выборочное={mom_g:.4f}, D выборочная={var_g:.7f}, D с известным m={d_theor_g:.7f}, M теор={m:.4f}, D теор={D2:.4f}")
print(
    f"Релеевское:    M выборочное={mom_r:.4f}, D выборочная={var_r:.7f}, D с известным m={d_theor_r:.7f}, M теор={mrter:.4f}, D теор={disprter:.4f}")
