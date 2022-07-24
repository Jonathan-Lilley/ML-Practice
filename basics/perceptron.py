import os
import numpy as np
from bigram_embeddings import Embeddings
from loss_functions import Loss
from activation_functions import Activation


class Perceptron:

    def __init__(self, languages, langdir,
                 loss_fn="mse", activation_fn="sigmoid", lr=0.1):
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

        self.loss_fn = Loss(loss_fn)
        self.activation_fn = Activation(activation_fn)
        self.W = self._initialize_weights()
        self.lr = lr
        self.b = self._initialize_bias()

    def _initialize_weights(self):
        return np.random.uniform(-1, 1, self.input_size)

    def _initialize_bias(self):
        return np.random.rand(1)

    def _loss_fn(self, tgt, pred):
        return self.loss_fn.loss(tgt, pred, self.train_data_count)

    def _d_loss_fn(self, tgt, pred):
        return self.loss_fn.d_loss(tgt, pred, self.train_data_count)

    def _activation(self, summed):
        return self.activation_fn.activation(summed)

    def _d_activation(self, summed):
        return self.activation_fn.derivative(summed)

    def _forward(self, in_arr):
        return self._activation((np.dot(self.W, in_arr)) + self.b)

    def _backward(self, sent, lang, out):
        d_W = sent
        d_activ = self._d_activation(np.dot(self.W, sent))
        d_loss = self._d_loss_fn(lang, out)
        return d_W, d_activ, d_loss

    def _step(self, d_W, d_activ, d_loss):
        self.b += d_activ * d_loss
        for ii, w in enumerate(self.W):
            gradient = d_W[ii] * d_activ * d_loss
            gradient = gradient * self.lr
            self.W[ii] += gradient

    def fit(self, epochs=1):
        for epoch in range(epochs):
            print(f"Epoch {epoch+1} of {epochs}")
            np.random.shuffle(self.train_ds)
            loss = 0
            for ii, (lang, sent) in enumerate(self.train_ds):
                out = self._forward(sent)
                loss += self._loss_fn(self.languages[lang], out)
                d_W, d_activ, d_loss = self._backward(sent,
                                                      self.languages[lang],
                                                      out)
                self._step(d_W, d_activ, d_loss)
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

