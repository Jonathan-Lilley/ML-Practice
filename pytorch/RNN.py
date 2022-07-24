import os
import torch
from torch import nn
from rec_bigram_embeddings import Embeddings
from pytorch_lookups import ACTIVATIONS


class LangClassRNN(nn.Module):

    def __init__(self, languages, langdir,
                 out_dims, hidden_size=10):
        super(LangClassRNN, self).__init__()

        if not hidden_size:
            raise "No hidden layer specified -- at least one required"

        self.languages = None
        self.val_to_lang = None
        self.lang_fs_train = None
        self.lang_fs_test = None
        self.embedder = None
        self.input_size = None
        self.train_ds = None
        self.train_data_count = None
        self.test_ds = None
        self.test_data_count = None

        self._generate_data(languages, langdir)

        self.U = None
        self.W = None
        self.V = None

        self._generate_weights(
            self.embedder.get_bigram_count,
            hidden_size, out_dims)

        self.h = torch.ones(hidden_size)
        self.b = nn.Parameter(torch.rand(hidden_size))

    def _generate_data(self, languages, langdir):
        self.languages = {lang: ii for ii, lang in enumerate(languages)}
        self.val_to_lang = {ii: lang for ii, lang in enumerate(languages)}

        self.lang_fs_train = [os.path.join(langdir, lang+"train.txt")
                              for lang in languages]
        self.lang_fs_test = [os.path.join(langdir, lang+"test.txt")
                             for lang in languages]
        self.embedder = Embeddings(self.lang_fs_train + self.lang_fs_test)

        self.input_size = self.embedder.get_bigram_count
        self.train_ds = []
        for lang, lang_f in zip(languages, self.lang_fs_train):
            self.train_ds += [(lang,
                               self.embedder.convert_sent_to_vect(line.strip()))
                              for line in open(lang_f) if line.strip()]
        self.train_ds = [(torch.tensor(self.languages[lang],
                                       dtype=torch.int64),
                          self._convert_to_tensor(sent))
                         for lang, sent in self.train_ds]
        self.train_data_count = len(self.train_ds)

        self.test_ds = []
        for lang, lang_f in zip(languages, self.lang_fs_test):
            self.test_ds += [(lang, self.embedder.convert_sent_to_vect(
                                      line.strip()))
                             for line in open(lang_f) if line.strip()]
        self.test_ds = [(torch.tensor(self.languages[lang],
                                      dtype=torch.int64),
                        self._convert_to_tensor(sent))
                        for lang, sent in self.test_ds]
        self.test_data_count = len(self.test_ds)

    def _generate_weights(self, in_size, hidden_size, out_size):
        self.U = nn.Parameter(torch.rand(in_size, hidden_size))
        self.W = nn.Parameter(torch.ones(hidden_size, hidden_size))
        self.V = nn.Parameter(torch.rand(hidden_size, out_size))

    def _convert_to_tensor(self, sent):
        return torch.from_numpy(sent)

    def forward(self, sentence):
        for bigram in sentence:
            a = torch.matmul(bigram, self.U) + torch.matmul(self.h, self.W) + self.b
            self.h = ACTIVATIONS["tanh"]()(a)
            self.h = self.h.detach()
            o = torch.matmul(self.h, self.V)
        yhat = ACTIVATIONS["softmax"](dim=0)(o)
        return yhat


class PTRNN(nn.Module):

    def __init__(self, languages, langdir,
                 out_dim, hidden_size=10):
        super(PTRNN, self).__init__()

        if not hidden_size:
            raise "No hidden layer specified -- at least one required"

        self.languages = None
        self.val_to_lang = None
        self.lang_fs_train = None
        self.lang_fs_test = None
        self.embedder = None
        self.input_size = None
        self.train_ds = None
        self.train_data_count = None
        self.test_ds = None
        self.test_data_count = None

        self._generate_data(languages, langdir)

        self.rnn = nn.RNN(self.embedder.get_bigram_count, hidden_size)
        self.linear = nn.Linear(hidden_size, out_dim)

        self.hidden = torch.ones(1, hidden_size)

    def _generate_data(self, languages, langdir):
        self.languages = {lang: ii for ii, lang in enumerate(languages)}
        self.val_to_lang = {ii: lang for ii, lang in enumerate(languages)}

        self.lang_fs_train = [os.path.join(langdir, lang+"train.txt")
                              for lang in languages]
        self.lang_fs_test = [os.path.join(langdir, lang+"test.txt")
                             for lang in languages]
        self.embedder = Embeddings(self.lang_fs_train + self.lang_fs_test)

        self.input_size = self.embedder.get_bigram_count
        self.train_ds = []
        for lang, lang_f in zip(languages, self.lang_fs_train):
            self.train_ds += [(lang,
                               self.embedder.convert_sent_to_vect(line.strip()))
                              for line in open(lang_f) if line.strip()]
        self.train_ds = [(torch.tensor(self.languages[lang],
                                       dtype=torch.int64),
                          self._convert_to_tensor(sent))
                         for lang, sent in self.train_ds]
        self.train_data_count = len(self.train_ds)

        self.test_ds = []
        for lang, lang_f in zip(languages, self.lang_fs_test):
            self.test_ds += [(lang, self.embedder.convert_sent_to_vect(
                                      line.strip()))
                             for line in open(lang_f) if line.strip()]
        self.test_ds = [(torch.tensor(self.languages[lang],
                                      dtype=torch.int64),
                        self._convert_to_tensor(sent))
                        for lang, sent in self.test_ds]
        self.test_data_count = len(self.test_ds)

    def _convert_to_tensor(self, sent):
        return torch.from_numpy(sent)

    def forward(self, input_values):
        for input_val in input_values:
            input_val = input_val.view(1, -1)
            a, h = self.rnn(input_val, self.hidden)
            self.hidden = h.detach()
        o = self.linear(a)
        o = o.flatten()
        yhat = ACTIVATIONS["softmax"](dim=0)(o)
        return yhat