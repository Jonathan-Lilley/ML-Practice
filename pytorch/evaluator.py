import torch
from pytorch_lookups import LOSSFNS


def bin_accuracy(pred, y):
    pred = pred > 0.5
    return pred == y

def softmax_accuracy(pred, y):
    pred = torch.argmax(pred)
    return pred == y

class Evaluator:

    def __init__(self, model, epochs=1, batch_size=1,
                 loss_fn="mse", device="cpu"):
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size

        self.device = device

        self.loss = LOSSFNS[loss_fn.lower()]

    def evaluate(self):
        with torch.no_grad():
            self.model.eval()
            corrects = 0
            for batch_n, (y, X) in enumerate(self.model.test_ds):
                pred = self.model(X)
                corrects += softmax_accuracy(pred, y)
            accuracy = corrects / self.model.test_data_count
            print(f"Binary accuracy: {accuracy}")

    def print_model_specs(self):
        print(self.model.layers)
        print(self.model.activations)