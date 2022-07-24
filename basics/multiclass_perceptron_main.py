from multiclass_perceptron import Perceptron


if __name__ == "__main__":
    langs = ["en", "de"]

    myperc = Perceptron(langs, "langdata", loss_fn="ce", lr=10.0, verbose=True)

    myperc.fit(epochs=10, epoch_size=0)
    myperc.evaluate()
