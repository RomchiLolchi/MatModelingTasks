# Задание 8 - Интегрирование
# Вариант 24: f(x) = x/sqrt(1+x^2); a=-3, b=1
# todo Лучше считать N везде по формулам
import math
from datetime import datetime
from numpy import random
import matplotlib.pyplot as plt
import numpy as np
from statistics import mean, stdev
from utils import user_input_or_default

a, b = -3.0, 1.0
def f(x): return x/np.sqrt(1+x**2)


xs = np.linspace(a, b, 400)
ys = f(xs)
plt.figure(figsize=(6,3))
plt.plot(xs, ys, linewidth=2)
plt.axhline(0, color='k', linewidth=0.6)
plt.title("График подынтегральной функции f(x)")
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
plt.tight_layout()
plt.show()


I_analytical = math.sqrt(1 + b ** 2) - math.sqrt(1 + a ** 2)
print(f"I по аналитическому решению = {I_analytical:.12f}")


tr_target_epsilon = 0.01
def trapezoid_integral(f, a, b, n):
    x = np.linspace(a, b, n+1)
    y = f(x)
    return (b-a)/n * (y[1:-1].sum() + ((y[0] + y[-1])/2))

maxN = 2000000
N = 1
I_analytical_abs = abs(I_analytical)
while True:
    I_tr = trapezoid_integral(f, a, b, N)
    tr_rel_err = abs(I_tr - I_analytical) / I_analytical_abs
    if tr_rel_err <= tr_target_epsilon or N > maxN:
        break
    N *= 2

if N > maxN:
    raise RuntimeError(f"Не удалось достичь требуемой точности метод трапеций при N<={maxN}")

low = N//2
high = N
while low + 1 < high:
    mid = (low + high)//2
    I_mid = trapezoid_integral(f, a, b, mid)
    if abs(I_mid - I_analytical)/I_analytical_abs <= tr_target_epsilon:
        high = mid
    else:
        low = mid

N_intervals_needed = high
nodes_needed = N_intervals_needed + 1
I_tr_final = trapezoid_integral(f,a,b,N_intervals_needed)
rel_err_tr = abs(I_tr_final - I_analytical) / I_analytical_abs
print(f"Метод трапеций:\n- Кол-во интервалов = {N_intervals_needed},\n- Кол-во узлов = {nodes_needed},\n- Интеграл = {I_tr_final:.12f},\n- Относительная ошибка = {rel_err_tr:.6f}")


def mc_uniform_once(f, a, b, N, rng):
    X = rng.uniform(a, b, N)
    return (b - a) * np.mean(f(X))

def mc_uniform_rep(f, a, b, N, runs=100, seed0=int(datetime.now().timestamp())):
    ss = np.random.SeedSequence(seed0)
    res = []
    for s in ss.spawn(runs):
        rng = np.random.default_rng(s)
        res.append(mc_uniform_once(f, a, b, N, rng))
    return np.array(res)

def mc_hit_or_miss_signed(f, a, b, N, seed=None):
    rng = np.random.default_rng(seed)
    X = rng.uniform(a, b, size=N)
    Y = rng.uniform(0.0, 1.0, size=N)

    xs = np.linspace(a, b, 10000)
    f_vals = f(xs)
    f_max_pos = max(0.0, f_vals.max())
    f_min_neg = min(0.0, f_vals.min())

    mask_pos = f(X) > 0
    Y_pos = rng.uniform(0, f_max_pos, size=N)
    hits_pos = np.sum(mask_pos & (Y_pos <= f(X)))

    mask_neg = f(X) < 0
    Y_neg = rng.uniform(0, -f_min_neg, size=N)
    hits_neg = np.sum(mask_neg & (Y_neg <= -f(X)))

    area_pos = (b - a) * f_max_pos
    area_neg = (b - a) * (-f_min_neg)

    I_est = (hits_pos / N) * area_pos - (hits_neg / N) * area_neg
    return I_est

def mc_hit_or_miss_signed_rep(f, a, b, N, runs=100, seed0=2):
    xs = np.linspace(a, b, 5000)
    fv = f(xs)
    f_min = fv.min()
    f_max = fv.max()

    ss = np.random.SeedSequence(seed0)
    res = []
    for s in ss.spawn(runs):
        rng = np.random.default_rng(s)
        res.append(mc_hit_or_miss_signed(f, a, b, N, rng))
    return np.array(res), f_min, f_max


N1 = user_input_or_default("Введите N1:", 2000)
ests1 = mc_uniform_rep(f, a, b, N1, runs=100, seed0=int(datetime.now().timestamp()))
mean1 = np.mean(ests1)
std1 = np.std(ests1, ddof=1)

print("Равномерный МК:")
print("- N =", N1)
print("- Интеграл =", mean1)
print("- Относит. ошибка =", abs(mean1 - I_analytical)/abs(I_analytical))
print("- Отклонение =", std1)

N2 = user_input_or_default("Введите N2:", 3000)
ests2, f_min, f_max = mc_hit_or_miss_signed_rep(f, a, b, N2, runs=100, seed0=int(datetime.now().timestamp()))
mean2 = np.mean(ests2)
std2 = np.std(ests2, ddof=1)

print("Второй вариант МК:")
print("- N =", N2)
print("- Интеграл =", mean2)
print("- Относит. ошибка =", abs(mean2 - I_analytical)/abs(I_analytical))
print("- Отклонение =", std2)