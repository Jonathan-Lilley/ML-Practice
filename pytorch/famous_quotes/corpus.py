import torch
from torch import nn

class Corpus(nn.Module):
    """Corpus for quotes
    
    Attributes:
        - quote list
        - author list
        - word list
        - SOQ and EOQ
        - hash tables 
            - word 2 idx
            - idx 2 word
            - auth 2 idx
            - idx 2 auth
    
    Methods:
        - word 2 idx lookup, idx 2 word lookup
        - auth 2 idx lookup
        - word to tensor
        - tensor to word
        - auth to tensor
        - quote to tensor
    """
    def __init__(self, file):
        super(Corpus, self).__init__()
        self.lines = [line.strip().split('\t')
                      for line in open(file) if line.strip()][::900]
        self.authors = list(set([line[0] for line in self.lines]))
        quotes = [line[1] for line in self.lines if line]

        words = set((' '.join(quotes)).split(' '))
        words.remove('')
        self.SOQ, self.EOQ = "<", ">"
        words.add(self.SOQ)
        words.add(self.EOQ)

        self.vocab = list(words)
        self.vocab_size = len(self.vocab)

        self.auth_size = len(self.authors)

        self.w2i = {word: idx for idx, word in enumerate(self.vocab)}
        self.i2w = {idx: word for idx, word in enumerate(self.vocab)}
        self.a2i = {auth: idx for idx, auth in enumerate(self.authors)}
    
    def w2t(self, word):
        word_tens = torch.zeros(self.vocab_size)
        idx = self.w2i[word]
        word_tens[idx] = 1
        return word_tens
    
    def t2w(self, tens):
        idx = int(torch.argmax(tens))
        word = self.i2w[idx]
        return word
    
    def a2t(self, auth):
        auth_tens = torch.zeros(self.auth_size)
        idx = self.a2i[auth]
        auth_tens[idx] = 1
        return auth_tens
    
    def quote_to_tensor(self, fullquote):
        auth, quote = fullquote
        #quote_tens = [self.a2t(auth)]
        quote_tens = [[self.w2t(self.SOQ)]
                      + [self.w2t(word) for word in quote.split(' ') if word]
                      + [self.w2t(self.EOQ)]]
        return quote_tens

    def idx_word(self, idx):
        return self.i2w[idx]

    @property
    def quotes(self):
        return self.lines

    @property
    def auth_count(self):
        return self.auth_size