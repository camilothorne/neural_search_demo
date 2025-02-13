import numpy as np
import faiss                  

class FaissIndex:

    def __init__(self, d:int)->None:

        '''
        Initialize the FaissIndex

        Args:
            d: dimension
        '''
        self.d = d
        self.index = faiss.IndexFlatL2(self.d)


    def update_index(self, xb:np.array)->None:
        '''
        Building the index
        Args:
            xb: vectors to be indexed
        '''
        self.index.add(xb) # add vectors to the index


    def is_non_empty(self)->None:
        '''
        Checks if index is trained / populated
        '''
        print(f'Index is trained? {self.index.is_trained}')


    def search(self, xq:np.array, k:int)->tuple[np.array, np.array]:
        '''
        Search for nearest neighbors
        Args:
            xq: queries
            k: number of nearest neighbors
        Returns:
            D: distances
            I: indices of the nearest neighbors
        '''
        D, I = self.index.search(xq, k)
        return D, I