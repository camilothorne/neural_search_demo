import pandas as pd
import numpy as np

class RRF:

    def __init__(self, ans1:pd.DataFrame, ans2:pd.DataFrame)->None:

        '''
        Initialize the RRF

        Args:
            ans1: answer set 1
            ans2: answer set 2
        '''
        self.ans1 = ans1[['doc_id']]
        self.ans2 = ans2[['doc_id']]
        self.ans1['rank1'] = self.ans1.index
        self.ans2['rank2'] = self.ans2.index
        self.ans = pd.merge(self.ans1, self.ans2, on=['doc_id'], how='outer').reset_index()
        self.N = float(self.ans.shape[0])
        self.ans['rank1'].fillna(self.N, inplace=True)
        self.ans['rank2'].fillna(self.N, inplace=True)
        self.ans['rerank'] = self.ans[['rank1','rank2']].apply(lambda x: self.compute_rrf(x[0], x[1]), axis=1)
        # Debug
        # print(f'RRF answer merge yielded an answer set of {self.ans.shape[0]} hits\n: {self.ans.head(10)}')
        self.ans.to_csv('results/rrf.csv', sep='\t', index=False)

    def get_rrf(self)->pd.DataFrame:
        '''
        Return the RRF dataframe
        '''
        return self.ans

    def compute_rrf(self, r1:float, r2:float, k=60)->float:
        '''
        Compute the RRF score

            rrf(d) = \sum_r [ 1 / (k + r(d)) ]

        Args:
            r1: rank of document in answer set 1
            r2: rank of document in answer set 2
            k: RRF constant
        Returns:
            RRF score
        '''
        rrf = ((1 / (k + r1)) + (1 / (k + r2)))
        return rrf