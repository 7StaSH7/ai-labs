import numpy as np

""" Реализация логической функции "И" (англ. "AND") с использованием нейрона Мак-Каллока-Питтса в биполярной логике"""


def main() -> None:
    print("Введите веса")
    w1 = float(input("Вес w1 = "))
    w2 = float(input("Вес w2 = "))
    print("Введите величину порога")
    theta = float(input("Порог = "))

    y = np.array([0, 0, 0, 0])
    
    x1 = np.array([1, 1, -1, -1])
    x2 = np.array([1, -1, 1, -1])
    
    z = np.array([-1, -1, -1, 1])

    con = 1

    while con != 0:
        zin = x1 * w1 + x2 * w2

        for i in range(0, 4):
            if zin[i] >= theta:
                y[i] = -1
            else:
                y[i] = 1

        print("Значение на выходе нейрона")
        print(y)

        if all(y == z):
            con = 0

        else:
            print(
                "Нейрон не обучен. Введите другие значения весовых коэффициентов и порога"
            )
            w1 = float(input("Вес w1 = "))
            w2 = float(input("Вес w2 = "))
            theta = float(input("Порог = "))

    print('Нейрон МакКаллока-Питса для функции "И" (англ. "AND")')
    print("Веса нейрона")
    print(w1, w2)

    print("Пороговое значение")
    print(theta)
