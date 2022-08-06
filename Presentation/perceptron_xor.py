import os
import numpy as np
from bigram_embeddings import Embeddings


class Perceptron:

    def __init__(self, lr=0.1):
        
        self.input_size = 2

        self.train_ds = [
            [0, [1, 1]],
            [1, [-1, 1]],
            [0, [-1, -1]],
            [1, [1, -1]]
        ]
        self.train_data_count = len(self.train_ds)

        self.W = self._initialize_weights()
        self.lr = lr

    def _initialize_weights(self):
        return np.random.uniform(-1, 1, self.input_size)

    def _forward(self, input_values):
        return np.dot(self.W, input_values)

    def _activation(self, value):
        if value > 0:
            return 1
        return 0

    def _step(self, input_values, pred, tgt):
        step_bool = np.array([tgt - pred], dtype=np.float32)
        self.W += self.lr * step_bool * input_values

    def fit(self, epochs=1):
        for epoch in range(epochs):
            print(f"Epoch {epoch+1} of {epochs}")
            np.random.shuffle(self.train_ds)
            loss = 0
            for ii, (class_val, inputs) in enumerate(self.train_ds):
                out = self._forward(inputs)
                pred = self._activation(out)
                loss += int(class_val != pred)
                self._step(inputs, pred, class_val)
            print(f"Loss: {loss/self.train_data_count}")

    def evaluate(self):
        correct = 0
        for class_val, inputs in self.train_ds:
            pred = self._activation(self._forward(inputs))
            if pred == class_val:
                correct += 1
        print(f"Total accuracy: {correct/self.train_data_count}")


if __name__ == "__main__":
    percp = Perceptron()

    for ii in range(20):
        percp.fit(epochs=10)
        percp.evaluate()
        print("\n")