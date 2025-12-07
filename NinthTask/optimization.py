# Задание 9 - Оптимизация
# Вариант 24: минимум f(x)=sin(x)/2x
import numpy as np
import matplotlib.pyplot as plt
from utils import user_input_or_default


def f(x): return np.sin(x) / (2 * x)


def plot_function(a, b):
    xs = np.linspace(a, b, 1000)
    ys = f(xs)

    plt.plot(xs, ys)
    plt.title("f(x) = sin(x) / (2x)")
    plt.grid(True)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.show()


def dichotomy(a, b, eps):
    iterations = 0
    delta = eps / 2

    while (b - a) > eps:
        xL = (a + b - delta) / 2
        xR = (a + b + delta) / 2
        yL, yR = f(xL), f(xR)

        if yL < yR:
            b = xR
        elif yL == yR:
            a = xL
            b = xR
        else:
            a = xL

        iterations += 1

    xm = (a + b) / 2
    ym = f(xm)
    return xm, ym, iterations


def golden_section(a, b, eps):
    tau = 1 /((1 + (5**0.5)) / 2)
    iterations = 0

    xL = b - tau * (b - a)
    xR = a + tau * (b - a)
    yL, yR = f(xL), f(xR)

    while (b - a) > eps:
        if yL < yR:
            b = xR
            xR = xL
            yR = yL
            xL = b - tau * (b - a)
            yL = f(xL)
        else:
            a = xL
            xL = xR
            yL = yR
            xR = a + tau * (b - a)
            yR = f(xR)

        iterations += 1

    xm = (a + b) / 2
    ym = f(xm)
    return xm, ym, iterations


def fibonacci_method(a, b, n): #, eps
    F = [1, 1]
    for i in range(2, n + 1):
        F.append(F[-1] + F[-2])

    iterations = 0

    xL = a + F[n - 2] / F[n] * (b - a)
    xR = a + F[n - 1] / F[n] * (b - a)
    yL, yR = f(xL), f(xR)

    for k in range(1, n):
        if yL < yR:
            b = xR
            xR = xL
            yR = yL
            xL = a + F[n - k - 2] / F[n - k] * (b - a)
            yL = f(xL)
        else:
            a = xL
            xL = xR
            yL = yR
            xR = a + F[n - k - 1] / F[n - k] * (b - a)
            yR = f(xR)

        # if (b - a) < eps:
        #     break
        iterations += 1

    xm = (a + b) / 2
    ym = f(xm)

    eps_pr = (b - a)
    return xm, ym, iterations, eps_pr


if __name__ == "__main__":
    plot_function(0.1, 10)

    a_int = float(user_input_or_default("Введите левую границу интервала:", 1))
    b_int = float(user_input_or_default("Введите правую границу интервала:", 6))
    eps_input = float(user_input_or_default("Введите точность:", 0.01))
    n_fib = int(user_input_or_default("Введите количество опытов по методу Фибоначчи:", 25))

    plot_function(a_int, b_int)

    print("\n=== Дихотомия ===")
    xm, ym, it = dichotomy(a_int, b_int, eps_input)
    print("Итерации:", it)
    print("Вычисления:", it * 2)
    print("x_min =", xm)
    print("f(x_min) =", ym)

    print("\n=== Золотое сечение ===")
    xm, ym, it = golden_section(a_int, b_int, eps_input)
    print("Итерации:", it)
    print("Вычисления:", it + 1)
    print("x_min =", xm)
    print("f(x_min) =", ym)

    print("\n=== Фибоначчи ===")
    xm, ym, it, eps_real = fibonacci_method(a_int, b_int, n_fib) #, eps_input
    print("Итерации:", it)
    print("Вычисления:", it + 1)
    print("x_min =", xm)
    print("f(x_min) =", ym)
    print("Точность на практике:", eps_real)
