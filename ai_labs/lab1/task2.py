import numpy as np
import matplotlib.pyplot as plt


def main() -> None:
    print("*** Программа вычислениЯ ДПФ гармонического сигнала ***")
    A = float(input("Введите амплитуду сигнала, ед.: "))
    f0 = float(input("Введите частоту сигнала, Гц: "))

    fdn = 2 * f0  # Частота дискретизации
    # в соответствии с критерием Найквиста
    mvis = 4
    fdv = mvis * fdn  # Частота дискретизации для визуализации
    dt = 1 / fdv  # Интервал дискретизации по времени
    T = 1 / f0  # Период сигнала
    NT = 6
    t = np.arange(0, NT * T, dt)  # Вектор времени, с
    y = A * np.sin(2 * np.pi * f0 * t)
    # Вектор сигнала
    N = len(y)

    # Дискретное преобразование Фурье
    k = np.arange(0, N)
    Ex = np.exp(-1j * 2 * np.pi / N * np.dot(np.transpose(k), k))
    Y = y * Ex

    # Обратное дискретное преобразование Фурье
    Ex = np.exp(1j * 2 * np.pi / N * np.dot(np.transpose(k), k))
    ys = Y / (N - 1) * Ex

    Y2 = Y * np.conj(Y)  # Квадрат модуля Фурье-образа
    ff = k * fdv / N  # Вектор частоты, Гц

    plt.figure()
    plt.plot(ff, Y2, "r")
    plt.xlabel("Frequency, Hz")
    plt.ylabel("Fourier-image modulus squared")

    _, ((ax1, ax2), (ax3, ax4)) = plt.subplots(figsize=(15, 4), nrows=2, ncols=2)

    ax1.plot(t, np.real(y), "r")
    ax1.set_xlim([0, NT * T])
    ax1.set_ylim([-1.1 * A, 1.1 * A])
    ax1.set_title("Real part")
    ax1.set_xlabel("Time, s")
    ax1.set_ylabel("Initial signal")

    ax2.plot(t, np.imag(y), "b")
    ax2.set_xlim([0, NT * T])
    ax2.set_ylim([-1.1 * A, 1.1 * A])
    ax2.set_title("Imaginary part")
    ax2.set_xlabel("Time, s")
    ax2.set_ylabel("Initial signal")

    ax3.plot(t, np.real(ys), "r")
    ax3.set_xlim([0, NT * T])
    ax3.set_ylim([-1.1 * A, 1.1 * A])
    ax3.set_xlabel("Time, s")
    ax3.set_ylabel("Restored signal")

    ax4.plot(t, np.imag(ys), "b")
    ax4.set_xlim([0, NT * T])
    ax4.set_ylim([-1.1 * A, 1.1 * A])
    ax4.set_xlabel("Time, s")
    ax4.set_ylabel("Restored signal")

    plt.show()

    print("**********   Конец работы   **********")
