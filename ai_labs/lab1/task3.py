import numpy as np
import matplotlib.pyplot as plt
from time import time


def main() -> None:
    t1 = float(input("Начало интервала: "))
    t2 = float(input("Конец интервала: "))
    N = float(input("Кол-во точек: "))

    def dft(*, _t1: float, _t2: float, _N: float, isPlot: bool = False) -> float:
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
            plt.title("DFT")
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

    def fft(*, _t1: float, _t2: float, _N: float, isPlot: bool = False) -> float:
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
            plt.title("FFT")
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

    def threshold(*, v, a, isPlot=True):
        x1 = list()
        x2 = list()
        y1 = list()
        y2 = list()

        for item in v:
            if item <= a:
                y1.append(0)
                x1.append(item)
            else:
                y2.append(1)
                x2.append(item)

        if isPlot:
            plt.axhline(0, color="black", linewidth=1)
            plt.axvline(0, color="black", linewidth=1)
            plt.plot(x1, y1, "-r")
            plt.plot(x2, y2, "-r")
            plt.grid()
            plt.show()

        return np.concatenate([y1, y2])

    def piecewise_linear(*, v, a0, a1, isPlot=True):
        y = list()
        x = list()
        tan_alpha = 1 / (a1 - a0)

        for item in v:
            if item <= a0:
                y.append(0)
                x.append(item)
            elif item >= a1:
                y.append(1)
                x.append(item)
            else:
                y.append(tan_alpha * (item - a1) + 1)
                x.append(item)

        if isPlot:
            plt.axhline(0, color="black", linewidth=1)
            plt.axvline(0, color="black", linewidth=1)
            plt.plot(x, y, "-r")
            plt.grid()
            plt.show()

        return y

    def sigmoid(*, v, a, isPlot=True):
        y = 1 / (1 + np.exp(-a * v))
        if isPlot:
            plt.axhline(0, color="black", linewidth=1)
            plt.axvline(0, color="black", linewidth=1)
            plt.plot(v, y, "-r")
            plt.grid()
            plt.show()

        return y

    def hyperbolic_tangent(*, v, a, isPlot=True):
        y = np.tanh(v / a)
        if isPlot:
            plt.plot(v, y, "-r")
            plt.axhline(0, color="black", linewidth=1)
            plt.axvline(0, color="black", linewidth=1)
            plt.grid()
            plt.show()

        return y

    dft(_t1=t1, _t2=t2, _N=N, isPlot=False)
    fft(_t1=t1, _t2=t2, _N=N, isPlot=False)

    dft_time = [dft(_t1=t1, _t2=t2, _N=2**i) for i in range(15, 20)]
    fft_time = [fft(_t1=t1, _t2=t2, _N=2**i) for i in range(15, 20)]

    plt.figure()
    plt.plot(list(range(15, 20)), dft_time, "-r", label="ДПФ")
    plt.plot(list(range(15, 20)), fft_time, "-b", label="БПФ")
    plt.legend()
    plt.grid(True)
    plt.xlabel("Data")
    plt.ylabel("Time, s")
    plt.show()

    v = np.arange(-8, 8, 0.05)

    threshold(v=v, a=3)
    piecewise_linear(v=v, a0=3, a1=6)
    sigmoid(v=v, a=1)
    hyperbolic_tangent(v=v, a=1)

    print("threshold: ")
    print(np.column_stack((v, threshold(v=v, a=3, isPlot=False))))

    print("piecewise linear: ")
    print(np.column_stack((v, piecewise_linear(v=v, a0=3, a1=6, isPlot=False))))

    print("sigmoid: ")
    print(np.column_stack((v, sigmoid(v=v, a=1, isPlot=False))))

    print("hyperbolic tangent: ")
    print(np.column_stack((v, hyperbolic_tangent(v=v, a=1, isPlot=False))))

    def diff(x, y):
        d = list()
        d.append(y[0])
        for i in range(0, len(x) - 1):
            d.append((y[i + 1] - y[i]) / (x[i + 1] - x[i]))
        return d

    result = diff(v, sigmoid(v=v, a=1, isPlot=False))

    plt.subplot(2, 1, 1)
    plt.plot(v, sigmoid(v=v, a=1, isPlot=False))
    plt.grid()
    plt.title("Sigmoid")
    plt.axhline(0, color="black", linewidth=1)
    plt.axvline(0, color="black", linewidth=1)

    plt.subplot(2, 1, 2)
    plt.plot(v, result)
    plt.grid()
    plt.title("Diff of sigmoid")
    plt.axhline(0, color="black", linewidth=1)
    plt.axvline(0, color="black", linewidth=1)

    plt.show()
