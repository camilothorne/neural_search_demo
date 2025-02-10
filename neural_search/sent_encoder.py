"""
This basic example loads a pre-trained model from the web and uses it to
generate sentence embeddings for a given list of sentences.
"""

import logging
import numpy as np
from sentence_transformers import SentenceTransformer
#from sentence_transformers import LoggingHandler

# #### Just some code to print debug information to stdout
# np.set_printoptions(threshold=100)

# logging.basicConfig(
#     format="%(asctime)s - %(message)s", 
#     datefmt="%Y-%m-%d %H:%M:%S",
#     level=logging.INFO, handlers=[LoggingHandler()]
# )

class SentEncode:

    def __init__(self, model_name="all-MiniLM-L6-v2"):

        # Load pre-trained Sentence Transformer Model. 
        # It will be downloaded automatically
        self.enc_model = SentenceTransformer(model_name)
        self.examples = [
        "This framework generates embeddings for each input sentence",
        "Sentences are passed as a list of string.",
        "The quick brown fox jumps over the lazy dog.",
        ]

    def embedd_text_batch(self, sentences):

        # Encode a batch of sentences
        return self.enc_model.encode(sentences)
    
    def embedd_text(self, sentence):

        # Encode a single sentence
        return self.enc_model.encode(sentence)

    def run_example(self):

        sentences = self.examples
        sentence_embeddings = self.embedd_text_batch(sentences)
        # The result is a list of sentence embeddings as numpy arrays
        for sentence, embedding in zip(sentences, sentence_embeddings):
            print("Sentence:", sentence)
            print("Embedding:", embedding)
            print("Embedding shape:", embedding.shape)
            print("")