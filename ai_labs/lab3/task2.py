import matplotlib.pyplot as plt
import numpy as np

from neurolab.net import newff
from neurolab.tool import minmax
from neurolab.trans import LogSig, PureLin
from ai_labs.lab3.task1 import postreg


def main() -> None:
    N = 100
    P = np.random.uniform(0, 1, (N, 21))
    T = np.random.uniform(0, 1, (3, N)).T

    model = newff(minmax(P), [21, 15, 3], [LogSig(), LogSig(), PureLin()])

    model.train(input=P, target=T, epochs=1000, goal=0.01)

    y = model.sim(np.random.uniform(0, 1, (N, 21)))

    postreg(title="Первые выходные данные", T=T[:, 0], y=y[:, 0])
    postreg(title="Вторые выходные данные", T=T[:, 1], y=y[:, 1])
    postreg(title="Третьи выходные данные", T=T[:, 2], y=y[:, 2])

    plt.show()
