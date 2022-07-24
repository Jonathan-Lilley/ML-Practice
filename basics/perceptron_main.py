from perceptron import Perceptron


if __name__ == "__main__":
    langs = ["en", "de"]

    myperc = Perceptron(langs, "langdata")

    myperc.fit(epochs=20)
    myperc.evaluate()
