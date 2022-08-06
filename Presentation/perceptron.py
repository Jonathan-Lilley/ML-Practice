import os
import numpy as np
from bigram_embeddings import Embeddings


class Perceptron:

    def __init__(self, languages, langdir, lr=0.1):
        self.languages = {lang: ii for ii, lang in enumerate(languages)}
        self.val_to_lang = {ii: lang for ii, lang in enumerate(languages)}

        self.lang_fs_train = None
        self.lang_fs_test = None
        self.embedder = None

        self.input_size = None
        self.train_ds = None
        self.train_data_count = None

        self.test_ds = None
        self.test_data_count = None

        self._gen_data(languages, langdir)

        self.W = self._initialize_weights()
        self.lr = lr

    def _gen_data(self, languages, langdir):

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


    def _initialize_weights(self):
        return np.random.uniform(-1, 1, self.input_size)

    def _forward(self, input_values):
        return np.dot(self.W, input_values)

    def _activation(self, value):
        if value > 0:
            return 1
        return 0

    def _step(self, input_values, pred, tgt):
        self.W += self.lr * (tgt - pred) * input_values
        #for w, _ in enumerate(self.W):
        #    self.W[w] += self.lr * (tgt - pred) * input_values[w]

    def fit(self, epochs=1):
        for epoch in range(epochs):
            print(f"Epoch {epoch+1} of {epochs}")
            np.random.shuffle(self.train_ds)
            loss = 0
            for ii, (lang, sent) in enumerate(self.train_ds):
                tgt = self.languages[lang]
                out = self._forward(sent)
                pred = self._activation(out)
                loss += int(tgt != pred)
                self._step(sent, pred, tgt)
            print(f"Loss: {loss/self.train_data_count}")

    def evaluate(self):
        correct = 0
        for lang, sent in self.test_ds:
            pred = self._activation(self._forward(sent))
            if pred == self.languages[lang]:
                correct += 1
        print(f"Total accuracy: {correct/self.test_data_count}")

