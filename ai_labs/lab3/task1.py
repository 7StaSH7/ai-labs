import matplotlib.pyplot as plt
import numpy as np

from neurolab.net import newff
from neurolab.tool import minmax
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
    plt.scatter(X_train, y_train, color="red", zorder=2.0)
    plt.plot(X_train, regressor.predict(X_train), color="blue")


def main() -> None:
    # Подготовка данных для обучения сети
    N = 100
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

    E = model.train(input=P, target=T, epochs=1000, goal=0.01)

    plt.figure()
    plt.plot(E)
    plt.xlabel("Эпохи")
    plt.ylabel("Отклонение")
    plt.grid()

    p = 0.2 * np.exp(-((x - 0.8) ** 2 / 0.7))
    
    Y = model.sim([p])
    print(Y)

    y = model.sim(P)

    postreg(title="P Первые выходные параметры", T=T[:, 0], y=y[:, 0])
    postreg(title="P Вторые выходные параметры", T=T[:, 1], y=y[:, 1])
    postreg(title="P Третьи выходные параметры", T=T[:, 2], y=y[:, 2])

    P1 = P + np.random.uniform(0, 0.01, P.shape)

    model1 = newff(minmax(P1), [21, 15, 3], [LogSig(), LogSig(), PureLin()])
    model1.train(input=P1, target=T, epochs=1000, goal=0.01)

    y1 = model1.sim(P1)

    postreg(title="P1 Первые выходные параметры", T=T[:, 0], y=y1[:, 0])
    postreg(title="P1 Вторые выходные параметры", T=T[:, 1], y=y1[:, 1])
    postreg(title="P1 Третьи выходные параметры", T=T[:, 2], y=y1[:, 2])

    P2 = P + np.random.uniform(0, 0.05, P.shape)

    model2 = newff(minmax(P2), [21, 15, 3], [LogSig(), LogSig(), PureLin()])
    model2.train(input=P2, target=T, epochs=1000, goal=0.01)

    y2 = model1.sim(P2)

    postreg(title="P2 Первые выходные параметры", T=T[:, 0], y=y2[:, 0])
    postreg(title="P2 Вторые выходные параметры", T=T[:, 1], y=y2[:, 1])
    postreg(title="P2 Третьи выходные параметры", T=T[:, 2], y=y2[:, 2])

    P3 = P + np.random.uniform(0, 0.1, P.shape)

    model3 = newff(minmax(P3), [21, 15, 3], [LogSig(), LogSig(), PureLin()])
    model3.train(input=P3, target=T, epochs=1000, goal=0.01)

    y3 = model1.sim(P3)

    postreg(title="P3 Первые выходные параметры", T=T[:, 0], y=y3[:, 0])
    postreg(title="P3 Вторые выходные параметры", T=T[:, 1], y=y3[:, 1])
    postreg(title="P3 Третьи выходные параметры", T=T[:, 2], y=y3[:, 2])

    P4 = P + np.random.uniform(0, 0.2, P.shape)

    model4 = newff(minmax(P4), [21, 15, 3], [LogSig(), LogSig(), PureLin()])
    model4.train(input=P4, target=T, epochs=1000, goal=0.01)

    y4 = model1.sim(P4)

    postreg(title="P4 Первые выходные параметры", T=T[:, 0], y=y4[:, 0])
    postreg(title="P4 Вторые выходные параметры", T=T[:, 1], y=y4[:, 1])
    postreg(title="P4 Третьи выходные параметры", T=T[:, 2], y=y4[:, 2])

    plt.show()
