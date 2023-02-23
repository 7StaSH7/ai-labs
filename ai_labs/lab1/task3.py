import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from time import time


def main() -> None:
    t1 = float(input("Начало интервала: "))
    t2 = float(input("Конец интервала: "))
    N = float(input("Кол-во точек: "))

    def dpf(*, _t1: float, _t2: float, _N: float, isPlot: bool) -> float:
        f0 = 2000
        mvis = 4
        fdn = 2 * f0  # Частота дискретизации
        fdv = mvis * fdn  # Частота дискретизации для визуализации

        t = np.arange(_t1, _t2, (_t2 - _t1) / _N)  # Вектор времени, с
        y = np.cos(2 * np.pi * f0 * t)  # Вектор сигнала

        exec_time = -time()

        # Дискретное преобразование Фурье
        k = np.arange(0, _N)
        Ex = np.exp(-1j * 2 * np.pi / _N * np.dot(np.transpose(k), k))
        Y = y * Ex

        exec_time = exec_time + time()

        # Обратное дискретное преобразование Фурье
        Ex = np.exp(1j * 2 * np.pi / _N * np.dot(np.transpose(k), k))
        ys = Y / (_N - 1) * Ex

        Y2 = Y * np.conj(Y)  # Квадрат модуля Фурье-образа
        ff = k * fdv / _N  # Вектор частоты, Гц

        if isPlot:
            plt.figure()
            plt.plot(ff, np.real(Y2), "-r")
            plt.xlabel("Frequency, Hz")
            plt.ylabel("Fourier-image modulus squared")

            plt.figure()
            plt.plot(t, np.real(y), "-r")
            plt.title("Real part")
            plt.xlabel("Time, s")
            plt.ylabel("Initial signal")

            plt.figure()
            plt.plot(t, np.imag(y), "-b")
            plt.title("Imaginary part")
            plt.xlabel("Time, s")
            plt.ylabel("Initial signal")

            plt.figure()
            plt.plot(t, np.real(ys), "-r")
            plt.title("Real part")
            plt.xlabel("Time, s")
            plt.ylabel("Restored signal")

            plt.figure()
            plt.plot(t, np.round(np.imag(ys), 5), "-b")
            plt.title("Imaginary part")
            plt.xlabel("Time, s")
            plt.ylabel("Restored signal")

            plt.show()

        return exec_time

    def bpf(*, _t1: float, _t2: float, _N: float, isPlot: bool) -> float:
        f0 = 2000
        mvis = 4
        fdn = 2 * f0  # Частота дискретизации
        fdv = mvis * fdn  # Частота дискретизации для визуализации

        t = np.arange(_t1, _t2, (_t2 - _t1) / _N)  # Вектор времени, с
        y = np.cos(2 * np.pi * f0 * t)  # Вектор сигнала

        exec_time = -time()

        # Быстрое преобразование Фурье
        Y = np.fft.fft(y)

        exec_time = exec_time + time()

        # Обратное дискретное преобразование Фурье
        ys = np.fft.ifft(Y)
        k = np.arange(0, _N)
        Y2 = Y * np.conj(Y)  # Квадрат модуля Фурье-образа
        ff = k * fdv / _N  # Вектор частоты, Гц

        if isPlot:
            plt.figure()
            plt.plot(ff, np.real(Y2), "-r")
            plt.xlabel("Frequency, Hz")
            plt.ylabel("Fourier-image modulus squared")

            plt.figure()
            plt.plot(t, np.real(y), "-r")
            plt.title("Real part")
            plt.xlabel("Time, s")
            plt.ylabel("Initial signal")

            plt.figure()
            plt.plot(t, np.imag(y), "-b")
            plt.title("Imaginary part")
            plt.xlabel("Time, s")
            plt.ylabel("Initial signal")

            plt.figure()
            plt.plot(t, np.real(ys), "-r")
            plt.title("Real part")
            plt.xlabel("Time, s")
            plt.ylabel("Restored signal")

            plt.figure()
            plt.plot(t, np.round(np.imag(ys), 5), "-b")
            plt.title("Imaginary part")
            plt.xlabel("Time, s")
            plt.ylabel("Restored signal")

            plt.show()

        return exec_time

    def sigmoid(*, _v: NDArray[np.floating], _a: int) -> None:
        pass

    dpf(_t1=t1, _t2=t2, _N=N, isPlot=False)
    bpf(_t1=t1, _t2=t2, _N=N, isPlot=False)

    dpf_time = [dpf(_t1=t1, _t2=t2, _N=2**i, isPlot=False) for i in range(15, 20)]
    bpf_time = [bpf(_t1=t1, _t2=t2, _N=2**i, isPlot=False) for i in range(15, 20)]
    
    plt.figure()
    plt.plot(list(range(15, 20)), dpf_time, "-r", label="ДПФ")
    plt.plot(list(range(15, 20)), bpf_time, "-b", label="БПФ")
    plt.legend()
    plt.grid(True)
    plt.xlabel('Data')
    plt.ylabel('Time, s')
    plt.show()

    v = np.arange(0, 10, 0.1)

    sigmoid(_v=v, _a=5)
