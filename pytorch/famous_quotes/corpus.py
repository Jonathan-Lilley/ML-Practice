import torch
from torch import nn

class Corpus(nn.Module):
    """Corpus for quotes"""
    def __init__(self, file, quote_embedding_dim):
        super(Corpus, self).__init__()
        self.lines = [line.strip().split('\t') for line in open(file)]
        authors, quotes = set([line[0] for line in self.lines]), \
                          [line[1] for line in self.lines]

        self.SOQ, self.EOQ = "<", ">"

        self.vocab = set((' '.join(quotes)).split(' '))
        self.vocab.add(self.SOQ)
        self.vocab.add(self.EOQ)
        self.vocab_size = len(self.vocab)

        self.word_to_idx = {word: ii for ii, word in enumerate(self.vocab)}
        self.idx_to_word = {ii: word for ii, word in enumerate(self.vocab)}

        self.author_to_idx = {author: ii for ii, author in enumerate(authors)}
        self.idx_to_author = {ii: author for ii, author in enumerate(authors)}

        #self.embeddings = nn.Embedding(self.vocab_size, quote_embedding_dim)
    

    def wto_tensor(self, word):
        idx = self.words["w2i"][word]
        tens = torch.zeros(self.vocab_size)
        tens[idx] = 1.
        return tens
    
    def wfrom_tensor(self, tens):
        idx = int(torch.nonzero(tens))
        return self.words["i2w"][idx]

    def ato_tensor(self, author):
        idx = self.words["a2i"][author]
        tens = torch.zeros(self.vocab_size)
        tens[idx] = 1.
        return tens

    def quote_to_tensor(self, quote):
        return (torch.tensor([self.words["w2i"][self.SOQ]] 
                          + [self.words["w2i"][w] for w in quote] 
                          + [self.words["w2i"][self.EOQ]]))


    @property
    def embeddings(self):
        return self.embeddings
    
    @property
    def words(self):
        return {"w2i": self.word_to_idx, "i2w": self.idx_to_word}
    
    @property
    def authors(self):
        return {"a2i": self.author_to_idx, "i2a": self.idx_to_author}
    
    @property
    def quotes(self):
        return self.lines