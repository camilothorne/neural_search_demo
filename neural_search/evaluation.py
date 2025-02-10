from scipy.stats import linregress
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

class EvaluateResults:

    def __init__(self, df_res:pd.DataFrame, fps:int)->None:
        '''
        Args:
            df_res: answer set data frame
            fps: # of relevant documents missing from df_res 
        '''     
        self.df_res = df_res
        self.fps = fps

    def compute_correlation(self, x:pd.Series, 
                            y:pd.Series)->tuple[float, float, float]:
        '''
        Computes correlation between two ranking scores
        '''
        slope_ow, _, r_value_ow, p_value_ow, _ = linregress(self.df_res.x, self.df_res.y)
        print(f'Correlation - slope: {slope_ow} r^2: {r_value_ow} p: {p_value_ow}')
        return slope_ow, r_value_ow, p_value_ow


    def recall_precision_f1_q(self, df_r:pd.DataFrame, k:int) -> dict:
        """
        Computes P@k, R@k and F1@k
        Args:
            df_ranking: answer data frame with two columns
            
                ranking relevant
                ...       ...
                doc_i     yes
                ...       ...
            
            k: top k prediction values for calculating metrics
        """
        rankings = df_r.head(k+1) # Restrict to top k
        tp = 0
        fp = 0
        fn = df_r[df_r['relevant']=='yes'].shape[0] + self.fps
        for _, row in rankings.iterrows():
            if row['relevant']=='yes':
                tp = tp + 1
                fn = fn - 1
            else:
                fp = fp + 1
        try:      
            p   = tp / (tp + fp)
        except:
            p   = 0
        try:
            r   = tp / (tp + fn)
        except:
            r   = 0
        try:
            f1  = (2 * p * r) / (p + r)
        except:
            f1  = 0
        res = {'precision': p, 'recall': r, 'f1-score': f1, 'top_k':str(k+1)}
        return res

    def compute_performance(self)->pd.DataFrame:
        '''
        This method computes metrics on ranking results
        '''
        perf = []
        for k in range(self.df_res.shape[0]):
            perf.append(self.recall_precision_f1_q(
                self.df_res[['scores', 
                             'relevant']].sort_values(
                                 by='scores', ascending=True),k))
        return pd.DataFrame(perf)