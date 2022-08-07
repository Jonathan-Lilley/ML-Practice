"""Generative Adversarial Network for Famous Quotes

Size notes:
W = word count

generator random = G
generator input = W
generator hidden = Hg
generator output = W

encoder input = W
encoder hidden = He
encoder output = D

discriminator input = D
discriminator hidden = Hd
discriminator output = 1
"""
import random
import torch
from torch import nn
from torch import optim
from generator import Generator
from discriminator import Discriminator
from encoder import Encoder
from corpus import Corpus
from fitter import fit_model as fit
import time

class GAN:
    """GAN put all together"""
    def __init__(self, corpus_file, generator_hidden_size,
                 encoder_hidden_size, enc_disc_size):
        self.corpus = Corpus(corpus_file)

        self.word_size = self.corpus.vocab_size

        self.generator = Generator(self.word_size, generator_hidden_size,
                                    self.word_size, self.corpus.auth_count)
        
        self.encoder = Encoder(self.word_size, encoder_hidden_size,
                               enc_disc_size)

        self.discriminator = Discriminator(enc_disc_size)

        self.gen_loss = nn.CrossEntropyLoss()
        self.disc_loss = nn.CrossEntropyLoss()

        self.optimizers = {
                    "gen": optim.RMSprop(self.generator.parameters()),
                    "enc": optim.RMSprop(self.encoder.parameters()),
                    "disc": optim.Adam(self.discriminator.parameters()),
        }

        self.quotes = self.corpus.quotes

        self.REAL, self.FAKE = 0, 1

        self.SOQ, self.EOQ = self.corpus.SOQ, self.corpus.EOQ

        print(f"Quote count: {len(self.corpus.quotes)}; "
                 f"word count: {self.word_size}")

    def generate(self, num_quotes=1000, teacher_forcing=False, decode=False):

        max_len = 25

        quote_tensors = []
        for quote in self.quotes:
            quote_tensors.append((torch.tensor(self.REAL, dtype=torch.long),
                                  self.corpus.quote_to_tensor(quote)[0]))
        
        gen_quotes = []
        for ii in range(num_quotes):
            print(f"Creating {ii+1} of {num_quotes} quotes")
            hidden = self.generator.generate_random()
            quote = []
            word = self.corpus.w2t(self.SOQ).clone().view(1, -1)
            #word = torch.unsqueeze(self.corpus.w2t(self.SOQ), 0)
            jj = 0
            while self.corpus.t2w(word) != self.EOQ:
                word, hidden = self.generator(word.clone(), hidden)
                quoted_word = word.clone()
                quote.append(quoted_word[0])
                jj += 1
                if teacher_forcing:
                    #word = torch.unsqueeze(quote_tensors[ii][1][jj], 0)
                    word = quote_tensors[ii][1][jj].clone().view(1, -1)
                else:
                    if jj > max_len:
                        break
            gen_quotes.append((torch.tensor(self.FAKE, dtype=torch.long),quote))

        if decode:
            self._decode(gen_quotes)

        else:
            quote_tensors += gen_quotes
            random.shuffle(quote_tensors)
            return quote_tensors

    def _decode(self, quotes):
        decoded = []
        for y, quote in quotes:
            quote_text = []
            for tens in quote:
                word = self.corpus.t2w(tens)
                quote_text.append(word)
            decoded.append(' '.join(quote_text))
        for quote in decoded:
            print(quote)

    def _encode(self, quote):
        hidden = self.encoder.initialize_hidden()
        for tens in quote:
            #tens = torch.unsqueeze(tens.clone(), 0)
            tens = tens.clone().view(1, -1)
            out, hidden = self.encoder(tens.clone(), hidden)
        return out

    def _discriminate(self, quote):     
        output = self.discriminator(quote.clone())[0]
        return output

    def fit(self, epochs=1):
        torch.autograd.set_detect_anomaly(True)
        for epoch in range(epochs):
            loss_ttl = 0
            print(f"Epoch {epoch+1} of {epochs}")
            quote_tensors = self.generate(num_quotes=5)
            for nn, (y, quote) in enumerate(quote_tensors):
                print(f"Item {nn+1} of {len(quote_tensors)}")
                for optimizer in self.optimizers.keys():
                    self.optimizers[optimizer].zero_grad()
                q = self._encode(quote)
                pred = self._discriminate(q)

                genloss = self.gen_loss(pred, y)
                genloss.backward(retain_graph=True)
                self.optimizers["gen"].step()

                disloss = self.disc_loss(pred, y)
                disloss.backward(retain_graph=True)
                self.optimizers["enc"].step()
                self.optimizers["disc"].step()

                loss_ttl += disloss.item()
                for optimizer in self.optimizers.keys():
                    self.optimizers[optimizer].step()
        print(f"Loss: {loss_ttl / len(quote_tensors)}")


if __name__ == "__main__":
    corp_file = "quotes.txt"
    gen_hidden = 10
    enc_hidden = 10
    enc_disc = 10
    myGAN = GAN(corp_file, gen_hidden, enc_hidden, enc_disc)

    #quotes = myGAN.generate(teacher_forcing=True, decode=True)

    myGAN.fit(epochs=5)
    myGAN.generate(num_quotes=3,decode=True)