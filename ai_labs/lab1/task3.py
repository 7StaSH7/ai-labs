import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from time import time


def task3() -> None:
    t1 = float(input("Начало интервала: "))
    t2 = float(input("Конец интервала: "))
    N = float(input("Кол-во точек: "))

    def dpf(*, t1: float, t2: float, N: float) -> float:
        pass

    def bpf(*, t1: float, t2: float, N: float) -> float:
        pass

    def sigmoid(*, v: NDArray[np.floating], a: int) -> None:
        pass

    v = np.arange(0, 10, 0.1)
    sigmoid(v, 5)
