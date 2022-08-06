#from perceptron import Perceptron
from nonlinear_perceptron import Perceptron
#from multilayer_perceptron import Perceptron

if __name__ == "__main__":
    langs = ["en", "de"]

    myperc = Perceptron(langs, "langdata", xor=True)

    myperc.fit(epochs=10)
    myperc.evaluate()
