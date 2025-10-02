# Задание 1 - вывод графика по точкам
import matplotlib.pyplot as plt
import numpy as np


def ask_file_and_get_datasets() -> list[list[list[float]]]:
    """
    Запрашивает у пользователя файл, читает все наборы данных из него и возвращает их

    Требования к файлу:
    Файл должен быть текстовым. На каждой строке помещается 2 значения по порядку (x и y), разделённые пробелом.
    В одном файле допускается хранение более 1 датасета, в таком случае датесеты разделяются пустыми строками

    :returns: Возвращает список, состоящий из списков. Структура проста: список, содержащий списки x и y (xy dataset pair) -> в каждом из его двух подсписков содержатся списки датасетов, в которых находятся значения x и y соответственно (т.е. [dataset pair: [x datasets: [1st x dataset: x_1, ..., x_n], ...], [y datasets: [1st y dataset: y_1, ..., y_n], ...]])
    """
    file = open(input("Введите путь до файла (относительный или абсолютный): "))
    line: str = file.readline()
    x_datasets: list[list[float]] = []
    y_datasets: list[list[float]] = []
    x_local_dataset: list[float] = []
    y_local_dataset: list[float] = []
    while line:
        if line == "\n":
            # Пустая строка - новый датасет
            x_datasets.append(x_local_dataset)
            x_local_dataset = []
            y_datasets.append(y_local_dataset)
            y_local_dataset = []
        else:
            # Не пустая строка - содержит x и y
            x, y = line.split(" ")
            try:
                # Сначала приводим к float, а потом добавляем в списки, чтобы сразу выявить возможную ошибку конвертации
                float_x = float(x)
                float_y = float(y)
                x_local_dataset.append(float_x)
                y_local_dataset.append(float_y)
            except TypeError:
                print(f"x или y содержат ошибку в строке '{line}', невозможно преобразовать к float")
        line = file.readline()

    x_datasets.append(x_local_dataset)
    y_datasets.append(y_local_dataset)

    print(f"Было найдено датасетов x y: {len(x_datasets)} {len(y_datasets)}")
    return [x_datasets, y_datasets]


def sort_dataset_pair_respectfully(xy_dataset_pair: list[list[float]]) -> list[list[float]]:
    """
    Функция, сортирующая пару списков: данные x и данные y. Сортировка учитывает связь x-y

    :param xy_dataset_pair: Список, состоящий из двух списков: первый - данные по x, второй - данные по y
    :return: Вывод такой же как и параметр xy_dataset_pair, но отсортированный
    """
    xy_dictionary = dict(zip(xy_dataset_pair[0], xy_dataset_pair[1]))
    xy_sorted_dictionary = dict(sorted(xy_dictionary.items()))
    return [list(xy_sorted_dictionary.keys()), list(xy_sorted_dictionary.values())]


def scatter(xy_datasets_pair: list[list[list[float]]]):
    """
    Выводит график из точек
    :param xy_datasets_pair: Список, содержащий списки x и y (xy dataset pair) -> в каждом из его двух подсписков содержатся списки датасетов, в которых находятся значения x и y соответственно (т.е. [dataset pair: [x datasets: [1st x dataset: x_1, ..., x_n], ...], [y datasets: [1st y dataset: y_1, ..., y_n], ...]])
    """
    fig, axes = plt.subplots()
    x_datasets, y_datasets = xy_datasets_pair
    assert len(x_datasets) == len(y_datasets), "Количество датасетов для отрисовки графика не совпадает!"
    for i in range(len(x_datasets)):
        current_x_dataset, current_y_dataset = x_datasets[i], y_datasets[i]
        axes.scatter(np.array(current_x_dataset), np.array(current_y_dataset))
    plt.show(block=True)


if __name__ == "__main__":
    global_datasets: list[list[list[float]]] = [[], []]

    # Файлы
    file_num = 1
    while True:
        if input(f"Добавить файл №{file_num}? [Y,n] ") == "n": break
        file_num += 1
        datasets = ask_file_and_get_datasets()
        for i in range(2):
            for j in datasets[i]:
                global_datasets[i].append(j)
    assert len(global_datasets[0]) == len(global_datasets[1]), "Количество датасетов не совпадает!"
    assert len(global_datasets[0]) > 0, "Количество датасетов <= 0!"

    # Сортировка
    for i in range(len(global_datasets[0])):
        global_datasets[0][i], global_datasets[1][i] = sort_dataset_pair_respectfully(
            [global_datasets[0][i], global_datasets[1][i]])

    # Выбор построения
    indices_answer = input(f"Обнаружено датасетов: {len(global_datasets[0])}. Введите через пробел индексы датасетов, которые нужно нарисовать на одном графике: (оставьте пустым для вывода всех)")
    indices_to_print: list[int]
    if indices_answer == "":
        indices_to_print = list(range(len(global_datasets[0])))
    else:
        indices_to_print = [int(i) for i in indices_answer.split(" ")]
    datasets_to_print: list[list[list[float]]] = [[], []]
    for i in indices_to_print:
        datasets_to_print[0].append(global_datasets[0][i])
        datasets_to_print[1].append(global_datasets[1][i])
    scatter(datasets_to_print)
