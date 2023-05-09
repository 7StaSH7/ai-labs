import matplotlib.pyplot as plt


def main() -> None:
    def dynamic_system(x, n, w) -> list:
        y = [x * w]
        for l in range(n):
            y.append(w * y[-1])
        return y

    plt.figure()
    plt.plot(range(21), dynamic_system(1, 20, 0.1))
    plt.grid()
    plt.title("|w| < 1")

    plt.figure()
    plt.plot(range(21), dynamic_system(1, 20, 1))
    plt.grid()
    plt.title("|w| = 1")

    plt.figure()
    plt.plot(range(21), dynamic_system(1, 20, 10))
    plt.grid()
    plt.title("|w| > 1")
    plt.show()
