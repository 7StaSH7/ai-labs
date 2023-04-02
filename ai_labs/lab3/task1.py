import matplotlib.pyplot as plt
import numpy as np
from neurolab.net import newff
from neurolab.tool import minmax
from neurolab.train import train_gd
from neurolab.trans import LogSig, PureLin
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def main() -> None:
    # Подготовка данных для обучения сети
    N = 1000
    P = np.zeros((N, 21))
    T = np.zeros((N, 3))
    x = np.linspace(0.05, 1.0, 21)

    for i in range(N):
        c = 0.9 * np.random.rand() + 0.1
        a = 0.9 * np.random.rand() + 0.1
        s = 0.9 * np.random.rand() + 0.1
        T[i, 0] = c
        T[i, 1] = a
        T[i, 2] = s
        P[i, :] = c * np.exp(-((x - a) ** 2) / s)

    # Размерность массивов
    print("P: ", P.shape)
    print("T: ", T.shape)
    print("x: ", x.shape)

    model = newff(minmax(P), [21, 15, 3], [LogSig(), LogSig(), PureLin()])

    E = model.train(input=P, target=T, epochs=1500, goal=0.01)

    plt.plot(E)
    plt.grid()

    y = model.sim(P)
    print(y)

    p = 0.2 * np.exp(-((x - 0.8) ** 2 / 0.7))
    Y = model.sim([p])
    print(Y)
