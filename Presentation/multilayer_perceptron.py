import os
import numpy as np
from bigram_embeddings import Embeddings
from loss_functions import Loss
from activation_functions import Activation


class Perceptron:

    def __init__(self, languages, langdir, h_layer_sizes=(5,3),
                 loss_fn="mse", activation_fn="sigmoid", lr=0.1, xor=False):
        if not xor:
            self.languages = {lang: ii for ii, lang in enumerate(languages)}
            self.val_to_lang = {ii: lang for ii, lang in enumerate(languages)}

            self.lang_fs_train = [os.path.join(langdir, lang+"train.txt")
                                for lang in languages]
            self.lang_fs_test = [os.path.join(langdir, lang+"test.txt")
                                for lang in languages]
            self.embedder = Embeddings(self.lang_fs_train + self.lang_fs_test)

            self.input_size = self.embedder.get_bigram_count()
            self.train_ds = []
            for lang, lang_f in zip(languages, self.lang_fs_train):
                self.train_ds += [(lang,
                                self.embedder.convert_sent_to_vect(line.strip()))
                                for line in open(lang_f) if line.strip()]
            self.train_data_count = len(self.train_ds)

            self.test_ds = []
            for lang, lang_f in zip(languages, self.lang_fs_test):
                self.test_ds += [(lang,
                                self.embedder.convert_sent_to_vect(line.strip()))
                                for line in open(lang_f) if line.strip()]
            self.test_data_count = len(self.test_ds)
        else:
            self.languages = {0: 0, 1: 1}
            self.val_to_lang = self.languages
            self.input_size = 2

            self.train_ds = [
                [0, [1, 1]],
                [1, [-1, 1]],
                [0, [-1, -1]],
                [1, [1, -1]]
            ]
            self.train_data_count = len(self.train_ds)

            self.test_ds = self.train_ds
            self.test_data_count = self.train_data_count

        self.h_layer_sizes = h_layer_sizes
        if not h_layer_sizes:
            self.h_layer_sizes = (100,)

        self.loss_fn = Loss(loss_fn)
        self.activation_fn = Activation(activation_fn)
        self.W = self._initialize_weights()
        self.lr = lr
        self.b = self._initialize_bias()

    def _initialize_weights(self):
        W = [np.random.uniform(-1, 1, (self.input_size, self.h_layer_sizes[0]))]
        W += [np.random.uniform(-1, 1, (size1, size2))
                      for size1, size2 in zip((self.h_layer_sizes),
                       self.h_layer_sizes[1:])]
        W += [np.random.uniform(-1, 1, (self.h_layer_sizes[-1], 1))]
        return W

    def _initialize_bias(self):
        b = [np.random.uniform(-1, 1, bsize) for bsize in self.h_layer_sizes]
        b += [np.random.uniform(-1, 1, 1)]
        return b

    def _loss_fn(self, tgt, pred):
        return self.loss_fn.loss(tgt, pred, self.train_data_count)

    def _d_loss_fn(self, tgt, pred):
        return self.loss_fn.d_loss(tgt, pred, self.train_data_count)

    def _activation(self, summed):
        return self.activation_fn.activation(summed)

    def _d_activation(self, summed):
        return self.activation_fn.derivative(summed)

    def _forward(self, in_arr):
        out = in_arr
        for Wi, b in zip(self.W, self.b): 
            out = np.dot(out, Wi) + b   
            out = self._activation(out)
        return out

    def _backward_graph(self, sent):
        outs = [np.array(sent)]
        for Wi, b in zip(self.W, self.b):
            outs.append(np.array(np.dot(outs[-1], Wi) + b))
        outs.reverse
        derivatives = [self._d_activation(outs[0])]
        for out in outs[1:]:
            derivative = self._d_activation(out)
            derivative += sum(derivatives[-1])
            derivatives.append(derivative)
        return derivatives[1:]

    def _backward(self, sent, lang, out_final):
        d_loss = self._d_loss_fn(lang, out_final)
        d_layers = self._backward_graph(sent)
        return d_layers, d_loss

    def _step(self, d_layers, d_loss):
        for bb, block in enumerate(d_layers):
            for nn, node in enumerate(block):
                self.b[bb] += node * d_loss
                self.W[bb] = self.W[nn] * node * d_loss

    def fit(self, epochs=1):
        for epoch in range(epochs):
            print(f"Epoch {epoch+1} of {epochs}")
            np.random.shuffle(self.train_ds)
            loss = 0
            for ii, (lang, sent) in enumerate(self.train_ds):
                out = self._forward(sent)
                loss += self._loss_fn(self.languages[lang], out)
                d_layers, d_loss = self._backward(sent,
                                                      self.languages[lang],
                                                      out)
                self._step(d_layers, d_loss)
            print(f"Loss: {loss/self.train_data_count}")

    def evaluate(self):
        correct = 0
        for lang, sent in self.test_ds:
            pred = self._forward(sent)
            if pred >= 0.5:
                pred = 1
            else:
                pred = 0
            if pred == self.languages[lang]:
                correct += 1
        print(f"Total accuracy: {correct/self.test_data_count}")

