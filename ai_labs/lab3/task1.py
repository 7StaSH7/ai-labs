import matplotlib.pyplot as plt
import numpy as np
from neurolab.net import newff
from neurolab.tool import minmax
from neurolab.train import train_gd
from neurolab.trans import LogSig, PureLin
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def postreg(*, title: str, T, y) -> None:
    X_train, X_test, y_train, y_test = train_test_split(
        y.reshape(-1, 1), T.reshape(-1, 1), test_size=1 / 3, random_state=0
    )
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    # Прогнозируем результаты тестовой выборки
    y_pred = regressor.predict(X_test)

    # Визуализация результатов тестового набор данных
    plt.figure()
    plt.title(title)
    plt.plot(X_test, y_test, color="blue")
    plt.scatter(X_test, y_pred, color="red")


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

    plt.figure()
    plt.plot(E)
    plt.grid()

    p = 0.2 * np.exp(-((x - 0.8) ** 2 / 0.7))
    Y = model.sim([p])
    print(Y)

    y = model.sim(P)

    postreg(title="Первые выходные параметры", T=T[:, 0], y=y[:, 0])
    postreg(title="Вторые выходные параметры", T=T[:, 1], y=y[:, 1])
    postreg(title="Третьи выходные параметры", T=T[:, 2], y=y[:, 2])

    P1 = P + np.random.uniform(0, 0.01, P.shape)

    plt.show()
