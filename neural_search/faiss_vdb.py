import numpy as np
import faiss                  

class FaissIndex:

    def __init__(self, d):

        '''
        Initialize the FaissIndex

        d: dimension
        '''
        self.d = d
        self.index = faiss.IndexFlatL2(self.d)


    def update_index(self, xb):
        '''
        Building the index
        '''
        self.index.add(xb) # add vectors to the index

    def is_non_empty(self):
        print(f'Index is trained? {self.index.is_trained}')


    def search(self, xq, k):
        '''
        Search for nearest neighbors

        xq: queries
        k: number of nearest neighbors
        '''
        D, I = self.index.search(xq, k)
        return D, I