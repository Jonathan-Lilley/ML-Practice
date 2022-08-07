import random
import torch
from torch import nn
from torch import optim


REAL, FAKE = 0, 1

def generate(corpus, generator, decoder, quotes):
    num_quotes = len(quotes)
    
    quote_tensors = []
    for quote in quotes:
        quote_tensors.append((REAL, corpus.quote_to_tensor(
                                    quote[1].split(' '))))

    generateds = []
    for ii in range(num_quotes):
        generateds.append(generator.forward())

    sentences = []
    for generated in generateds:
        word = corpus.wto_tensor(corpus.SOQ)
        sentence = [word]
        hidden = torch.unsqueeze(generated, 0)
        while corpus.wfrom_tensor(word) != corpus.EOQ:
            print(corpus.wfrom_tensor(word))
            print(word.shape)
            word, hidden = decoder(word, hidden)
            sentence.append(word)
        sentences.append(sentence)

    for sentence in sentences:
        quote_tensors.append((FAKE, corpus.quote_to_tensor(sentence)))

    return quote_tensors


def discriminate(encoder, discriminator, quote):

    hidden = encoder.initialize_hidden()
    for w in quote:
        o, hidden = encoder(w, hidden)
    encoded = o

    pred = discriminator(encoded)

    return pred
     

def fit_model(corpus, generator, decoder, encoder, discriminator, epochs=1):

    loss_fn = nn.CrossEntropyLoss()

    optimizers = {
                #"embed": optim.Adam(corpus.parameters()),
                "gen": optim.Adam(generator.parameters()),
                "dec": optim.RMSprop(decoder.parameters()), 
                "enc": optim.RMSprop(encoder.parameters()),
                "disc": optim.Adam(discriminator.parameters()),
    }

    quotes = corpus.quotes

    for epoch in range(epochs):
        print(f"Epoch {epoch+1} of {epochs}")

        loss_total = 0

        quote_tensors = generate(corpus, generator, decoder, quotes)
        
        random.shuffle(quote_tensors)

        for y, quote in quote_tensors:

            corpus.zero_grad()
            generator.zero_grad()
            decoder.zero_grad()
            encoder.zero_grad()
            discriminator.zero_grad()
            
            pred = discriminate(encoder, discriminator, quote)

            loss = loss_fn(pred, y)

            loss.backward()

            loss_total += loss.item()

            for optimizer in optimizers:
                optimizers[optimizer].step()
    
    print(f"Loss: {loss_total / len(quote_tensors)}")
            