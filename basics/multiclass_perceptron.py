import os
import time
import numpy as np
from bigram_embeddings import Embeddings
from loss_functions import Loss
from activation_functions import Activation


class Perceptron:

    def __init__(self, languages, langdir,
                 loss_fn="mse", activation_fn="softmax", lr=0.1,
                 verbose=False):
        self.verbose = verbose
        self.languages = {lang: ii+1 for ii, lang in enumerate(languages)}
        self.val_to_lang = {ii+1: lang for ii, lang in enumerate(languages)}

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
        self.train_ds = np.array(self.train_ds, dtype=object)
        self.train_data_count = len(self.train_ds)

        self.test_ds = []
        for lang, lang_f in zip(languages, self.lang_fs_test):
            self.test_ds += [(lang,
                              self.embedder.convert_sent_to_vect(line.strip()))
                             for line in open(lang_f) if line.strip()]
        self.test_ds = np.array(self.test_ds, dtype=object)
        self.test_data_count = len(self.test_ds)

        if self.verbose:
            print(f"Data count\nTrain data: {self.train_data_count}\n"
                  f"Test data: {self.test_data_count}\n")

        self.loss_fn = Loss(loss_fn)
        self.activation_fn = Activation(activation_fn)
        self.W = self._initialize_weights()
        self.b = self._initialize_bias()
        self.lr = lr

    def _initialize_weights(self):
        return np.random.uniform(-1, 1, (len(self.languages), self.input_size))

    def _initialize_bias(self):
        return np.random.uniform(-1, 1, len(self.languages))

    def _loss_fn(self, tgt, pred):
        return self.loss_fn.loss(tgt, pred)

    def _d_loss_fn(self, tgt, pred):
        return self.loss_fn.d_loss(tgt, pred)

    def _activation(self, summed):
        return self.activation_fn.activation(summed)

    def _d_activation(self, summed):
        return self.activation_fn.derivative(summed)

    def _forward(self, sent):
        return self._activation(np.dot(self.W, sent) + self.b)

    def _backward(self, sent, lang, out):
        d_W = sent
        d_activ = self._d_activation(np.dot(self.W, sent) + self.b)
        d_loss = self._d_loss_fn(lang, out)
        print("dactive\n", d_activ)
        print("dloss\n", d_loss)
        input()
        return d_W, d_activ, d_loss

    def _step(self, d_W, d_activ, d_loss):
        for class_idx, vals in enumerate(zip(self.b, self.W)):
            b, w = vals
            self.b[class_idx] += d_activ[class_idx] * d_loss[class_idx]
            for ii, w_ii in enumerate(w):
                gradient = ((d_W[ii]
                             * d_activ[class_idx]
                             * d_loss[class_idx])
                            * self.lr)
                self.W[class_idx][ii] += gradient

    def fit(self, epochs=1, epoch_size=1):
        if epoch_size <=0 or epoch_size > self.train_data_count:
            epoch_size = self.train_data_count
        if self.verbose:
            start = time.time()
            print(f"Running {epochs} epochs.")
        for epoch in range(epochs):
            print(f"Epoch {epoch+1} of {epochs}")
            #np.random.shuffle(self.train_ds)
            sentenceidxs = np.random.choice(len(self.train_ds), size=epoch_size)
            sentences = self.train_ds[sentenceidxs]
            loss = 0
            for ii, (lang, sent) in enumerate(sentences):
                out = self._forward(sent)
                loss += self._loss_fn(self.languages[lang], out)
                d_W, d_activ, d_loss = self._backward(sent,
                                                      self.languages[lang],
                                                      out)
                self._step(d_W, d_activ, d_loss)
                if self.verbose:
                    current = time.time()
                    print(f"Sample {ii+1} of {epoch_size} || "
                          f"Elapsed time: {round(current - start, 1)}",
                          end='\r')
            print("\n")
            current = time.time()
            print(f"epoch time: {round(current - start, 1)}\n"
                  f"Loss:\n{loss/self.train_data_count}\n")

    def evaluate(self):
        correct = 0
        for lang, sent in self.test_ds:
            pred = self._forward(sent)
            outval = np.argmax(pred)
            if outval == self.languages[lang]:
                correct += 1
        print(f"Total accuracy: {correct/self.test_data_count}")

