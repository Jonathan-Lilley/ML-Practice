"""Generative Adversarial Network for Famous Quotes

Size notes:
generator input = I
generator outputs = H
decoder input = W
decoder hidden = H
decoder output = W
encoder input = W
encoder hidden = H
encoder output = J
discriminator input = J
"""
from generator import Generator
from discriminator import Discriminator
from encoder import Encoder
from decoder import Decoder
from corpus import Corpus
from fitter import fit_model as fit


if __name__ == "__main__":
    quote_embedding_dim = 100
    corp = Corpus("quotes.txt", quote_embedding_dim)

    rand_size = 100
    generator_hidden = 100
    generator_output = 100
    auth_count = len(corp.authors["a2i"])

    decoder_input = len(corp.words["w2i"])
    decoder_hidden = generator_output
    decoder_output = decoder_input

    encoder_input = decoder_output
    encoder_hidden = decoder_hidden
    encoder_output = 100
    
    discriminator_input = encoder_output

    quotes_generator = Generator(rand_size, generator_hidden,
                                 generator_output, auth_count)
    
    quotes_decoder = Decoder(decoder_input, decoder_hidden, decoder_output)

    quotes_encoder = Encoder(encoder_input, encoder_hidden, encoder_output)

    quotes_discriminator = Discriminator(encoder_output)

    fit(corp, quotes_generator, quotes_decoder,
         quotes_encoder, quotes_discriminator)