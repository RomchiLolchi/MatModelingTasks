# Задание 5 - численные методы решений задачи Коши дифференциальных уравнений
# Вариант 24: a=0,6; b=0,6; l=1,3; n=2
# x в [0,2]; h = 0,2; y'=(-1)^n*a*y+b*x^2+l; y(0)=1
#todo Возможно надо расширить количество знаков после запятой (для расчёта погрешности)
import math
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate


def equation(xn, yn):
    return 0.6 * yn + 0.6 * xn ** 2 + 1.3


def analytical_answer(x_param):
    return 157 / 18 * math.exp(0.6 * x_param) - x_param ** 2 - 10 / 3 * x_param - 139 / 18


def euler_method(f, xn, yn, h):
    return yn + f(xn, yn) * h


def runge_kutta_method_4_deg(f, xn, yn, h):
    k1 = f(xn, yn)
    k2 = f(xn + h / 2, yn + k1 * h / 2)
    k3 = f(xn + h / 2, yn + k2 * h / 2)
    k4 = f(xn + h, yn + h * k3)
    return yn + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


if __name__ == "__main__":
    x = np.arange(0, 2.2, 0.2)
    y_a = np.array([analytical_answer(x_val) for x_val in x])

    yn_e_temp = 1
    yn_rk_temp = 1
    y_e_temp = [1]
    y_rk_temp = [1]
    for i in range(1, len(x)):
        x_prev = x[i - 1]
        e = euler_method(equation, x_prev, y_e_temp[-1], 0.2)
        rk = runge_kutta_method_4_deg(equation, x_prev, y_rk_temp[-1], 0.2)
        y_e_temp.append(e)
        y_rk_temp.append(rk)
    y_e = np.array(y_e_temp)
    y_rk = np.array(y_rk_temp)

    print("Таблица значений:")
    print(tabulate([
        ["xi"] + [str(r) for r in x],
        ["y аналитический"] + [str(r) for r in y_a],
        ["y по методу Эйлера"] + [str(r) for r in y_e],
        ["y по методу Рунге-Кутты"] + [str(r) for r in y_rk]
    ]))

    fig, ax = plt.subplots()
    ax.plot(x, y_a, label="y аналитический", marker="o")
    ax.plot(x, y_e, label="y по методу Эйлера", marker="v")
    ax.plot(x, y_rk, label="y по методу Рунге-Кутты", marker="s")
    ax.legend()
    plt.show()
