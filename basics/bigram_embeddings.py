import os
import numpy as np


class Embeddings:

    def __init__(self, files):
        self.lang_fs = files
        self.characters = ['<SOS>', '<EOS>']
        self.bigrams = self._extract_all_bigrams()
        self.bigram_count = len(self.bigrams)

    def _extract_all_bigrams(self):
        self.characters += self._extract_all_characters()
        return [(ch1, ch2) for ch1 in self.characters
                   for ch2 in self.characters]

    def _extract_all_characters(self):
        return list(set(''.join([''.join([line.strip() for line in open(langf)])
                                for langf in self.lang_fs])))

    def convert_sent_to_vect(self, sentence):
        sent_bigrams = [('<SOS>', sentence[0])] \
                           + list(zip(sentence, sentence[1:])) \
                           + [(sentence[-1], '<EOS>')]
        sent_vect = np.zeros(self.bigram_count)
        for bigram in sent_bigrams:
            sent_vect[self.bigrams.index(bigram)] += 1
        return sent_vect

    def get_bigram_count(self):
        return self.bigram_count