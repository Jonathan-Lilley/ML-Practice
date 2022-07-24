import os
import torch
from torch import nn
from bigram_embeddings import Embeddings
from pytorch_lookups import ACTIVATIONS


class LangClasserFFNN(nn.Module):

    def __init__(self, languages, langdir,
                 out_dims, layer_activations, hidden_sizes=(10,)):
        super(LangClasserFFNN, self).__init__()

        if not hidden_sizes:
            raise "No hidden layer specified -- at least one required"

        if len(layer_activations) - len(hidden_sizes) != 1:
            raise "Number of layers and layer activations must be equal, with" \
                  " one extra activation for the final layer"

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

        self.layers = nn.ModuleList(self._generate_layer_networks(
            out_dims, hidden_sizes))

        self.activations = self._generate_layer_activations(layer_activations)

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

    def _generate_layer_networks(self, out_dims, hidden_sizes):
        layers = [nn.Linear(self.embedder.get_bigram_count,
                           hidden_sizes[0])]
        for layer_count, layer_size in enumerate(hidden_sizes[1:]):
            layers.append(nn.Linear(hidden_sizes[layer_count-1],
                                         hidden_sizes[layer_count]))
        layers.append(nn.Linear(hidden_sizes[-1], out_dims))
        return layers

    def _generate_layer_activations(self, activations):
        layer_activations = [ACTIVATIONS[activation.lower()]()
                                 for activation in
                                 activations]
        return layer_activations

    def _gen_batches(self, batch_size):
        pass

    def _convert_to_tensor(self, sent):
        return torch.from_numpy(sent)

    def forward(self, sent_tens):
        out = sent_tens
        for layer, activation in zip(self.layers, self.activations):
            out = activation(layer(out))
        return out