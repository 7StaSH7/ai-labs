import numpy as np
import matplotlib.pyplot as plt


def main() -> None:
    print("*** Программа визуализации гармонического сигнала ***")

    f0 = float(input("Введите частоту сигнала в герцах: "))
    A = float(input("Введите амплитуду сигнала в условных единицах: "))
    phi = float(input("Введите фазу сигнала, рад: "))
    NT = float(input("Введите количество периодов наблюдения, ед.: "))
    mvis = float(input("Введите коэффициент уменьшения интервала дискретизации: "))

    print("-----------------------------------------------------")
    print("Введенные данные:")
    print("Частота сигнала (Гц)                           = ", f0)
    print("Амплитуда сигнала (усл.ед.)                    = ", A)
    print("Фаза сигнала, рад                              = ", phi)
    print("Количество периодов сигнала                    = ", NT)
    print("Коэффициент уменьшения интервала дискретизации = ", mvis)
    print("-----------------------------------------------------")

    w = 2 * np.pi * f0  # Переход к круговой частоте (рад/сек)
    T = 1 / f0  # Период исходного сигнала (сек)
    dtn = np.pi / w  # Интервал дискретизации Найквиста (теорема Котельникова)
    dtv = dtn / mvis  # Интервал дискретизации, пригодный для визуализации

    Tnab = NT * T  # Интервал наблюдения (сек)

    print("Интервал наблюдения = ", Tnab, " сек")

    Nn = (
        Tnab / dtn
    )  # Количество точек в интервале наблюдения для дискретизации Найквиста
    Nv = (
        Tnab / dtv
    )  # Количество точек в интервале наблюдения для интервала дискретизации,
    # пригодного для визуализации

    print("Количество точек в интервале наблюдения для дискретизации Найквиста = ", Nn)
    print(
        "Количество точек в интервале наблюдения для интервала дискретизации, пригодного для визуализации, = ",
        Nv,
    )

    tn = dtn * np.arange(0, Nn - 1)
    # Вектор времени (сек), дискретизированный
    # в соответствии с критерием Найквиста

    tv = dtv * np.arange(0, Nv - 1)
    # и c интервалом дискретизации,
    # пригодным для визуализации

    y1n = A * np.cos(2 * np.pi * f0 * tn + phi)
    y2n = A * np.sin(2 * np.pi * f0 * tn + phi)
    # Вектора сигналов, дискретизированных
    # в соответствии с критерием Найквиста

    y1v = A * np.cos(2 * np.pi * f0 * tv + phi)
    y2v = A * np.sin(2 * np.pi * f0 * tv + phi)
    # Вектора сигналов, дискретизированных
    # c интервалом дискретизации,
    # пригодным для визуализации

    _, ((ax1, ax2), (ax3, ax4)) = plt.subplots(figsize=(15, 4), nrows=2, ncols=2)
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    ax1.plot(tn, y1n, "r-*")
    ax1.set_xlim([0, Tnab])
    ax1.set_ylim([-1.1 * A, 1.1 * A])
    ax1.title.set_text("Дискретизация по Найквисту")
    ax1.set_xlabel("Time, sec")
    ax1.set_ylabel("Signal")

    ax2.plot(tv, y1v, "b")
    ax2.set_xlim([0, Tnab])
    ax2.set_ylim([-1.1 * A, 1.1 * A])
    ax2.set_title(f"Дискретизация в {mvis} раз точнее")
    ax2.set_xlabel("Time, sec")
    ax2.set_xlabel("Signal")

    ax3.plot(tn, y2n, "r-*")
    ax3.set_xlim([0, Tnab])
    ax3.set_ylim([-1.1 * A, 1.1 * A])
    ax3.set_xlabel("Time, sec")
    ax3.set_ylabel("Signal")

    ax4.plot(tv, y2v, "b")
    ax4.set_xlim([0, Tnab])
    ax4.set_ylim([-1.1 * A, 1.1 * A])
    ax4.set_xlabel("Time, sec")
    ax4.set_ylabel("Signal")

    plt.show()
