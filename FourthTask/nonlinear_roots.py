# Задание 4 - решение нелинейных уравнений
# Вариант 24: a(b + x) = 2^x; при коэф. -1, 0: f(x) = -x-2^x, 1 корень (x≈−0.64)
#todo пересмотреть хорды; не останавливается решение
#todo через метод Ньютона почему-то кол-во шагов больше, чем нужно
import math
import matplotlib.pyplot as plt
import numpy as np
import sys

a, b = -1, 0
a_coef, b_coef = -1, 0
x0 = -1
epsilon = sys.float_info.epsilon
N = 100


def function_test(x):
    return x ** 2 + 4 * x - 5


def function(x):
    global a_coef, b_coef
    return a_coef * (b_coef + x) - 2 ** x


def dichotomy(f, a, b, epsilon, nmax, n=1):
    x0 = (a + b) / 2
    if f(x0) == 0 or (b - a) < 2 * epsilon or n >= nmax:
        print(f"Решение найдено за {n} итераций")
        return x0
    if f(a) * f(x0) > 0:
        return dichotomy(f, x0, b, epsilon, nmax, n + 1)
    else:
        return dichotomy(f, a, x0, epsilon, nmax, n + 1)


def fd_test(x):
    return 2 * x + 4


def sd_test(x):
    return 2


def first_derivative(x):
    global a_coef, b_coef
    return a_coef - 2 ** x * math.log(2)


def second_derivative(x):
    global a_coef, b_coef
    return -2 ** x * (math.log(2)) ** 2


def chord_method(f, fd, sd, xn, a, b, epsilon, nmax, n=1):
    if fd(xn) > 0 and sd(xn) > 0 and f(a) < 0 and f(b) > 0:
        xnew = xn - (b - xn) * f(xn) / (f(b) - f(xn))
    elif fd(xn) < 0 and sd(xn) < 0 and f(a) > 0 and f(b) < 0:
        xnew = xn - (b - xn) * f(xn) / (f(b) - f(xn))
    elif fd(xn) > 0 and sd(xn) < 0 and f(a) < 0 and f(b) > 0:
        xnew = xn - (xn - a) * f(xn) / (f(xn) - f(b))
    elif fd(xn) < 0 and sd(xn) > 0 and f(a) > 0 and f(b) < 0:
        xnew = xn - (xn - a) * f(xn) / (f(xn) - f(b))
    else:
        raise RuntimeError()

    if f(xnew) == 0 or math.fabs(xnew - xn) < epsilon or n >= nmax:
        print(f"Решение найдено за {n} итераций")
        return xnew
    if f(xn) * f(b) < 0:
        return chord_method(f, fd, sd, xn, xn, b, epsilon, nmax, n + 1)
    else:
        return chord_method(f, fd, sd, xn, a, xn, epsilon, nmax, n + 1)


def get_test_phi_functions():
    def phi1(x):
        if -4 * x + 5 < 0:
            return x
        return -math.sqrt(-4 * x + 5)

    return phi1


def get_phi_functions():
    # Способ 1: через логарифм
    def phi1(x):
        if a_coef * (b_coef + x) <= 0:
            return x
        return math.log2(a_coef * (b_coef + x))

    # Способ 2: добавление x
    def phi2(x):
        return a_coef * (b_coef + x) - 2 ** x + x

    return phi1, phi2


def check_convergence(phi, x0, a, b, points=1000):
    def derivative(f, x, h=1e-7):
        return (f(x + h) - f(x - h)) / (2 * h)

    x_values = np.linspace(a, b, points)
    max_derivative = max(abs(derivative(phi, x)) for x in x_values)

    if max_derivative < 1:
        print("Условие сходимости выполняется")
        return True
    else:
        print("Условие сходимости не выполняется")
        return False


def simple_iterations(phi, a, b, x0, epsilon, n_max):
    check_convergence(phi, x0, a, b)

    def recursive_iteration(x, iter_count=1):
        if iter_count >= n_max:
            print(f"Достигнут лимит итераций ({n_max})")
            return x

        x_next = phi(x)

        if abs(x_next - x) < epsilon:
            print(f"Решение найдено за {iter_count} итераций")
            return x_next

        return recursive_iteration(x_next, iter_count + 1)

    return recursive_iteration(x0)


def newton_method(f, df, x0, epsilon, n_max):
    def recursive_newton(x, iter_count=1):
        if iter_count >= n_max:
            print(f"Достигнут лимит итераций ({n_max})")
            return x

        fx = f(x)
        dfx = df(x)

        if abs(fx) < epsilon:
            print(f"Решение найдено за {iter_count} итераций")
            return x

        if abs(dfx) < epsilon:
            print(f"Производная близка к нулю в точке x = {x}")
            return x

        x_next = x - 0.5 * fx / dfx

        if abs(x_next - x) > 1e10:
            print("Метод расходится")
            return x

        return recursive_newton(x_next, iter_count + 1)

    return recursive_newton(x0)


if __name__ == "__main__":
    test_input = int(input("Тестовый режим? x**2+4*x-5=0; x1=-5; x2=1; локализация первого корня; 1 - да, 0 - нет: "))
    if test_input == 1:
        ab_input = input(f"Введите границы отрезка в виде a b, либо оставьте без ввода для -6 -4: ")
        if ab_input != "":
            split = ab_input.split(" ")
            a, b = float(split[0]), float(split[1])
        else:
            a, b = -6, -4
        x0_input = input(f"Введите начальное приближение x0, либо оставьте без ввода для -5.8: ")
        if x0_input != "":
            x0 = float(x0_input)
        else:
            x0 = -5.8
        epsilon_input = input(f"Введите точность epsilon, либо оставьте без ввода для {epsilon}: ")
        if epsilon_input != "":
            epsilon = float(epsilon_input)
        n_input = input(f"Введите максимальное количество итераций, либо оставьте без ввода для {N}: ")
        if n_input != "":
            N = int(n_input)

        print("Вывод графика")
        fig, axes = plt.subplots()
        x = np.linspace(-10, 10)
        y = np.array([function_test(x_val) for x_val in x])
        axes.plot(x, [0 for _ in x])
        axes.plot(x, y)
        plt.show()

        method = int(input(r"""
                ╔ Выберите метод:
                ╠ 1 - Половинного деления (дихотомии/бисекции)
                ╠ 2 - Хорд
                ╠ 3 - Простой итерации
                ╚ 4 - Ньютона

                """))
        if method == 1:
            print(f"Метод дихотомии вернул корень: {dichotomy(function_test, a, b, epsilon, N)}")
        elif method == 2:
            print(
                f"Метод хорд вернул корень: {chord_method(function_test, fd_test, sd_test, x0, a, b, epsilon, N)}")
        elif method == 3:
            print(
                f"Метод простой итерации вернул корень: {simple_iterations(get_test_phi_functions(), a, b, x0, epsilon, N)}")
        else:
            print(f"Метод Ньютона вернул корень: {newton_method(function_test, fd_test, x0, epsilon, N)}")

    else:
        ab_coef_input = input(f"Введите коэффициенты в виде a b, либо оставьте без ввода для {a_coef} {b_coef}: ")
        if ab_coef_input != "":
            split = ab_coef_input.split(" ")
            a_coef, b_coef = int(split[0]), int(split[1])
        ab_input = input(f"Введите границы отрезка в виде a b, либо оставьте без ввода для {a} {b}: ")
        if ab_input != "":
            split = ab_input.split(" ")
            a, b = float(split[0]), float(split[1])
        x0_input = input(f"Введите начальное приближение x0, либо оставьте без ввода для {x0}: ")
        if x0_input != "":
            x0 = float(x0_input)
        epsilon_input = input(f"Введите точность epsilon, либо оставьте без ввода для {epsilon}: ")
        if epsilon_input != "":
            epsilon = float(epsilon_input)
        n_input = input(f"Введите максимальное количество итераций, либо оставьте без ввода для {N}: ")
        if n_input != "":
            N = int(n_input)

        print("Вывод графика; a = -1, b = 0; x≈−0.64")
        fig, axes = plt.subplots()
        x = np.linspace(-5, 5)
        y = np.array([function(x_val) for x_val in x])
        axes.plot(x, [0 for _ in x])
        axes.plot(x, y)
        plt.show()

        method = int(input(r"""
        ╔ Выберите метод:
        ╠ 1 - Половинного деления (дихотомии/бисекции)
        ╠ 2 - Хорд
        ╠ 3 - Простой итерации
        ╚ 4 - Ньютона
        
        """))
        if method == 1:
            print(f"Метод дихотомии вернул корень: {dichotomy(function, a, b, epsilon, N)}")
        elif method == 2:
            print(
                f"Метод хорд вернул корень: {chord_method(function, first_derivative, second_derivative, x0, a, b, epsilon, N)}")
        elif method == 3:
            print(
                f"Метод простой итерации вернул корень: {simple_iterations(get_phi_functions()[1], a, b, x0, epsilon, N)}")
        else:
            print(f"Метод Ньютона вернул корень: {newton_method(function, first_derivative, x0, epsilon, N)}")
