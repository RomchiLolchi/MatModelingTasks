# Задание 7 - МНК (аппроксимация методом наименьших квадратов; методы регрессионного анализа)
import numpy as np
import FirstTask.dot_plotting as dp
import math
import matplotlib.pyplot as plt
from tabulate import tabulate


def get_good_coeffs(x, y, degree):
    N = len(x)
    m = degree

    A = [[0.0 for _ in range(m + 1)] for __ in range(m + 1)]
    B = [0.0 for _ in range(m + 1)]

    for k in range(N):
        xi = x[k]
        yi = y[k]
        for i in range(m + 1):
            for j in range(m + 1):
                A[i][j] += xi ** (i + j)
            B[i] += yi * (xi ** i)

    n = m + 1
    for i in range(n):
        diag = A[i][i]
        for j in range(i, n):
            A[i][j] /= diag
        B[i] /= diag
        for r in range(n):
            if r == i:
                continue
            factor = A[r][i]
            for j in range(i, n):
                A[r][j] -= factor * A[i][j]
            B[r] -= factor * B[i]
    return B


def horner(poly_coeffs, x):
    m = len(poly_coeffs) - 1
    result = poly_coeffs[m]
    for i in range(m - 1, -1, -1):
        result = result * x + poly_coeffs[i]
    return result


if __name__ == "__main__":
    global_datasets: list[list[list[float]]] = dp.ask_user_for_datasets()
    dp.sort_global_dataset(global_datasets)

    dataset_no = int(input("Выберите датасет для вывода: (с 0) "))
    N = len(global_datasets[0][dataset_no])
    x = global_datasets[0][dataset_no]
    y = global_datasets[1][dataset_no]

    degree = int(input(f"Введите степень аппроксимирующего полинома: "))
    if degree < 0 or degree >= N:
        raise RuntimeError("Ошибка: степень должна быть >= 0 и < N")

    coeffs = get_good_coeffs(x, y, degree)

    print(f"\nКоэффициенты полинома (a0 ... a{degree}):")
    for i, c in enumerate(coeffs):
        print(f"a[{i}] = {c}")

    guesses = list()
    errors = list()
    sse = 0.0
    for xi, yi in zip(x, y):
        guess = horner(coeffs, xi)
        error = yi - guess

        guesses.append(guess)
        errors.append(error)

        sse += error**2

    print("\nТаблица:")
    print(tabulate([
        ["x_i"] + [str(r) for r in x],
        ["y_y"] + [str(r) for r in y],
        ["f(x_i)"] + [str(r) for r in guesses],
        ["ошибка (y_i - f(x_i))"] + [str(r) for r in errors]
    ]))

    print(f"\nСуммарная квадратичная ошибка: {sse}")

    plt.scatter(x, y)
    plt.plot(x, guesses)
    plt.show()