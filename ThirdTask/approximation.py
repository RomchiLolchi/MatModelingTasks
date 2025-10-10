# Задание 2 - аппроксимация
import scipy
import FirstTask.dot_plotting as dp
import numpy as np

def piecewise_linear_interpolation(dataset: list[list[float]], subpoints: int) -> list[list[float]]:
    assert len(dataset[0]) == len(dataset[1]), "Длина датасетов x y не совпадает!"
    new_dataset: list[list[float]] = [[], []]

    new_dataset[0].append(dataset[0][0])
    new_dataset[1].append(dataset[1][0])

    for i in range(1, len(dataset[0])):
        a = (dataset[1][i] - dataset[1][i - 1]) / (dataset[0][i] - dataset[0][i - 1])
        b = dataset[1][i - 1] - a * dataset[0][i - 1]
        for j in range(subpoints):
            delta = abs((dataset[0][i - 1] - dataset[0][i]) / subpoints)
            newx = new_dataset[0][len(new_dataset[0]) - 1] + delta
            newy = a * newx + b
            new_dataset[0].append(newx)
            new_dataset[1].append(newy)
    new_dataset[0].append(dataset[0][-1])
    new_dataset[1].append(dataset[1][-1])
    return new_dataset


def piecewise_parabolic_interpolation(dataset: list[list[float]], subpoints: int) -> list[list[float]]:
    assert len(dataset[0]) == len(dataset[1]), "Длина датасетов x y не совпадает!"
    new_dataset: list[list[float]] = [[], []]

    new_dataset[0].append(dataset[0][0])
    new_dataset[1].append(dataset[1][0])

    for i in range(1, len(dataset[0]) - 1):
        matrix_x = np.array(
            [[dataset[0][i - 1] ** 2, dataset[0][i - 1], 1], [dataset[0][i] ** 2, dataset[0][i], 1],
             [dataset[0][i + 1] ** 2, dataset[0][i + 1], 1]])
        matrix_y = np.array([dataset[1][i - 1], dataset[1][i], dataset[1][i + 1]])
        a, b, c = np.linalg.solve(matrix_x, matrix_y)

        for j in range(int(subpoints)):
            delta: float
            if i == len(dataset[0]) - 2:
                delta = abs((dataset[0][i + 1] - dataset[0][i - 1]) / subpoints)
            else:
                delta = abs((dataset[0][i - 1] - dataset[0][i]) / subpoints)
            newx = new_dataset[0][-1] + delta
            newy = a * newx ** 2 + b * newx + c
            new_dataset[0].append(newx)
            new_dataset[1].append(newy)
        new_dataset[0].append(dataset[0][i])
        new_dataset[1].append(dataset[1][i])
    new_dataset[0].append(dataset[0][-2])
    new_dataset[1].append(dataset[1][-2])
    new_dataset[0].append(dataset[0][-1])
    new_dataset[1].append(dataset[1][-1])

    return new_dataset


def lagrange_polynomial_interpolation(dataset: list[list[float]], subpoints: int, n: int = -1) -> list[
    list[float]]:
    assert len(dataset[0]) == len(dataset[1]), "Длина датасетов x y не совпадает!"
    if n != -1:
        assert n <= len(dataset[0]) - 1, "n не должно быть > (кол-ва точек графика - 1)!"
    else:
        n = len(dataset[0]) - 1
    assert n >= 2, "n должно быть >= 2"

    new_dataset: list[list[float]] = [[], []]

    new_dataset[0].append(dataset[0][0])
    new_dataset[1].append(dataset[1][0])

    for i in range(int(np.ceil(len(dataset[0]) / n))):
        new_points = [dataset[0][i:i + n + 1], dataset[1][i:i + n + 1]]
        for j in range(subpoints):
            delta = abs((new_points[0][0] - new_points[0][-1]) / subpoints)
            newx = new_dataset[0][-1] + delta
            newy = lagrange_get_l(new_points, newx)
            new_dataset[0].append(newx)
            new_dataset[1].append(newy)
        new_dataset[0].append(dataset[0][i])
        new_dataset[1].append(dataset[1][i])

    new_dataset[0].append(dataset[0][-1])
    new_dataset[1].append(dataset[1][-1])

    return new_dataset


def lagrange_get_l(dataset: list[list[float]], x) -> float:
    if len(dataset[0]) == 2:
        return (1 / (dataset[0][1] - dataset[0][0])) * np.linalg.det(
            [[x - dataset[0][0], dataset[1][0]], [x - dataset[0][1], dataset[1][1]]])
    else:
        dataset_no_latest = [dataset[0][:-1], dataset[1][:-1]]
        dataset_no_first = [dataset[0][1:], dataset[1][1:]]
        return (1 / (dataset[0][-1] - dataset[0][0])) * np.linalg.det(
            [[x - dataset[0][0], lagrange_get_l(dataset_no_latest, x)],
             [x - dataset[0][-1], lagrange_get_l(dataset_no_first, x)]])


def spline_interpolation(dataset: list[list[float]], subpoints: int) -> list[list[float]]:
    assert len(dataset[0]) == len(dataset[1]), "Длина датасетов x y не совпадает!"
    new_dataset: list[list[float]] = [[], []]
    spline = scipy.interpolate.CubicSpline(dataset[0], dataset[1])

    new_dataset[0] = list(np.arange(stop=dataset[0][-1], start=dataset[0][0], step=abs(dataset[0][-1]-dataset[0][0])/subpoints))
    for x in new_dataset[0]:
        new_dataset[1].append(spline(x))

    return new_dataset

if __name__ == "__main__":
    global_datasets: list[list[list[float]]] = dp.ask_user_for_datasets()
    dp.sort_global_dataset(global_datasets)

    strpoints_amount = input("Введите количество интерполяционных точек: (оставьте пустым для стандартного значения) ")
    points_amount: int
    if strpoints_amount == "":
        points_amount = 100
    else:
        points_amount = int(strpoints_amount)

    chosen_method = int(input(r"""
    ╔ Выберите интерполяционный метод:
    ╠ 1 - Кусочно-линейный
    ╠ 2 - Кусочно-параболический
    ╠ 3 - Полиномы Лагранжа
    ╚ 4 - Кубическое сплайн интерполирование
    
    """))
    if chosen_method == 1:
        for i in range(len(global_datasets[0])):
            global_datasets[0][i], global_datasets[1][i] = piecewise_linear_interpolation(
                [global_datasets[0][i], global_datasets[1][i]], subpoints=points_amount)
    elif chosen_method == 2:
        for i in range(len(global_datasets[0])):
            global_datasets[0][i], global_datasets[1][i] = piecewise_parabolic_interpolation(
                [global_datasets[0][i], global_datasets[1][i]], subpoints=points_amount)
    elif chosen_method == 3:
        strn = input("Введите степень полинома: (оставьте пустым для стандартного n-1 значения, где n - число точек) ")
        n = -1
        if strn != "":
            n = int(strn)

        for i in range(len(global_datasets[0])):
            global_datasets[0][i], global_datasets[1][i] = lagrange_polynomial_interpolation(
                [global_datasets[0][i], global_datasets[1][i]], subpoints=points_amount, n=n)
    elif chosen_method == 4:
        for i in range(len(global_datasets[0])):
            global_datasets[0][i], global_datasets[1][i] = spline_interpolation(
                [global_datasets[0][i], global_datasets[1][i]], subpoints=points_amount)

    dp.choose_datasets_and_scatter(global_datasets)
