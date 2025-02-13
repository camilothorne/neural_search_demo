import bm25s
import numpy as np

class BM25Index:

    def __init__(self)->None:
        '''
        Initialize the index
        '''
        self.index = bm25s.BM25(backend="auto")

    def update_index(self, corpus:list)->None:
        '''
        Building the index
        Args:
            corpus: list of documents to be indexed
        '''
        corpus_tokens = bm25s.tokenize(corpus, stopwords="en")
        self.index.index(corpus_tokens) # index documents

    def search(self, query:str, k:int)->tuple[np.array, np.array]:
        '''
        Search for nearest neighbors
        Args:
            q: query
            k: number of nearest neighbors
        Returns:
            D: BM25 scores
            I: indices of the nearest neighbors
        '''
        query_tokens = bm25s.tokenize(query, stopwords="en")
        D, I = self.index.retrieve(query_tokens, k=k)
        return D, I