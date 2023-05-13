import matplotlib.pyplot as plt
import numpy as np

from .digits import digits, eight, five, four, nine, one, seven, six, three, two, zero


def main() -> None:
    def weights_matrix(N, x) -> list:
        num_samples = len(x)
        Mmax = N / (2 * np.log(N))

        if num_samples > Mmax:
            return "Количество чисел превышает Mmax"

        W = np.zeros((N, N))

        for sample in x:
            if len(sample) != N:
                return "Размерность числа не соответствует числу нейронов"

            xi = np.array(sample).reshape(N, 1)
            W += np.dot(xi, xi.T)

        np.fill_diagonal(W, 0)

        return W

    def weights_matrix_with_Mmax(N, x, Mmax) -> list:
        W = np.zeros((N, N))

        for sample in x:
            if len(sample) != N:
                return "Размерность числа не соответствует числу нейронов"

            xi = np.array(sample).reshape(N, 1)
            W += np.dot(xi, xi.T)

        np.fill_diagonal(W, 0)

        return W

    def train_hop_net(_digits) -> list:
        n, m = len(_digits), len(_digits[0])
        arr = np.ndarray((n, m))
        for i in range(n):
            for j in range(m):
                arr[i, j] = _digits[i][j]

        return np.dot(arr.T, arr)

    def test_hop_net(weights, digit, iterations) -> list:
        new_digit = np.array(digit).copy()

        for epoch in range(iterations):
            update_order = np.random.permutation(range(63))

            for i in range(63):
                neuron = update_order[i]
                net = np.dot(new_digit, weights[:, neuron])
                new_digit[neuron] = np.sign(net)

        return new_digit

    def noisy_digit(digit, noise):
        if noise < 0 or noise > 1:
            print("Шум должен быть в пределах от 0 до 1")
        return digit + np.random.uniform(-noise, noise, np.asarray(digit).size)

    def visualize_number(number):
        if type(number) != np.ndarray:
            number = np.asarray(number)
        plt.figure()
        plt.imshow(number.reshape(9, 7), cmap="Greys")

    N = 3
    x = np.array([[1, -1, 1]])
    w = weights_matrix(N, x)
    print(w)

    N = 3
    x = np.array([[-1, 1, -1], [1, -1, 1]])
    Mmax = 10
    w = weights_matrix_with_Mmax(N, x, Mmax)
    print(w)

    w1 = train_hop_net([zero])
    print(w1)

    w = train_hop_net([zero, one])
    visualize_number(test_hop_net(w, zero, 1))
    visualize_number(test_hop_net(w, one, 1))

    w = train_hop_net([two, three, four])
    visualize_number(test_hop_net(w, two, 1))
    visualize_number(test_hop_net(w, three, 1))
    visualize_number(test_hop_net(w, four, 1))

    w = train_hop_net([five, six])
    visualize_number(test_hop_net(w, five, 1))
    visualize_number(test_hop_net(w, six, 1))

    w = train_hop_net([seven, eight, nine])
    visualize_number(test_hop_net(w, seven, 1))
    visualize_number(test_hop_net(w, eight, 1))
    visualize_number(test_hop_net(w, nine, 1))

    visualize_number(test_hop_net(w, noisy_digit(seven, 0.3), 1))
    visualize_number(test_hop_net(w, noisy_digit(eight, 0.3), 1))
    visualize_number(test_hop_net(w, noisy_digit(nine, 0.3), 1))
    plt.show()
