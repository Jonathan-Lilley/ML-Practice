from FFNN import LangClasserFFNN
from RNN import LangClassRNN, PTRNN
from fitter import Fitter
from evaluator import Evaluator


if __name__ == "__main__":
    epochs = 10
    languages = ["de", "en", "eo"]
    datadir = "langdata"
    loss = "ce"
    if loss == "ce":
        out_dims = len(languages)
    else:
        out_dims = 3
    activations = ("relu", "relu", "softmax")
    hidden_size = 10
    classifier = PTRNN(languages, datadir, out_dims, hidden_size)

    fitter = Fitter(classifier, loss_fn="ce", epochs=epochs)
    fitter.fit()

    evaluator = Evaluator(classifier)

    evaluator.evaluate()
