import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neural_network as nn
from sklearn.metrics import confusion_matrix
import os


def main() -> None:
    def plotchar(i):
        plt.figure()
        plt.imshow(i.reshape(7, 5))

    def max_error(model: nn.MLPClassifier, train: np.ndarray, err: int):
        confusion = np.zeros((32, 32))
        for _ in range(1000):
            confusion += confusion_matrix(
                target.argmax(axis=1),
                model.predict(train + np.random.normal(0, err, train.shape)).argmax(
                    axis=1
                ),
            )
        return 1 - (np.min(np.diag(confusion)) / 1000.0)

    alphabet = pd.read_csv(os.path.join(os.path.dirname(__file__), "alphabet.csv"))

    vovels = alphabet[
        alphabet["letter"].isin(["А", "Е", "И", "О", "У", "Ы", "Э", "Ю", "Я"])
    ].drop("letter", axis=1)

    plotchar(vovels.values[0])

    train = alphabet.drop("letter", axis=1).values
    target = np.eye(32)

    model = nn.MLPClassifier(
        hidden_layer_sizes=(10,),
        activation="logistic",
        max_iter=500000,
        alpha=1e-4,
        solver="sgd",
        tol=0.01,
        random_state=1,
        learning_rate_init=0.1,
        n_iter_no_change=10000,
    )

    model.fit(train, target)

    error = 0.3

    train_with_noise = train + np.random.normal(0, error, train.shape)

    plotchar(train_with_noise[0, :])

    upgraded_model = nn.MLPClassifier(
        hidden_layer_sizes=(20, 20),
        activation="logistic",
        max_iter=500000,
        alpha=1e-4,
        solver="sgd",
        tol=0.01,
        random_state=1,
        learning_rate_init=0.1,
        n_iter_no_change=10000,
    )

    upgraded_model.fit(train, target)

    noise_array = np.arange(0, 0.5, 0.05)

    model_noise = [max_error(model, train, noise) for noise in noise_array]
    upgraded_model_noise = [
        max_error(upgraded_model, train, noise) for noise in noise_array
    ]

    plt.figure()
    plt.plot(noise_array, model_noise)
    plt.plot(noise_array, upgraded_model_noise)
    plt.grid()
    plt.xlabel("Шум")
    plt.ylabel("Ошибка")
    plt.legend(["Двухслойный перцептрон", "Трехслойный перцептрон"])

    plt.show()
