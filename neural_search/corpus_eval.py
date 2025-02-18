import pandas as pd
import time


from preprocessing.sent_encoder import SentEncode
from neural_search.faiss_vdb import FaissIndex
from neural_search.bm25_vdb import BM25Index
from neural_search.query_eval import *


def corpus_eval_bm25(test_data:pd.DataFrame, 
                     merged_data:pd.DataFrame,
                     k:int,
                     ind_map:dict,
                     ddb:BM25Index)->pd.DataFrame:
    '''
    Run all queries on idex and measure:

        micro-averaged p@i, r@i, f1@i, for 1 =< i =< k
    
    '''
    print('\n==> Evaluation across test set (BM25)...\n')
    
    begin = time.time()
    df_q = test_data[['query', 'query_id', 'query_docs']].drop_duplicates(keep='first')
    
    print(f'* Shape of test set (queries):\t{df_q.shape}')
    print(f'* Test set snapshot:\n{df_q.sort_values(by="query_docs").head(20)}\n')
    
    results = []
    for _, row in df_q.iterrows():
        _, eval_result = query_eval_bm25(row['query'], 
                                    merged_data, 
                                    k, 
                                    ind_map, 
                                    ddb)
        results.append(eval_result)
    df_c = pd.concat(results)
    df_res = df_c.groupby(level=0).mean(numeric_only=True)
    end = time.time()
    
    print(f'* Time taken: {end-begin:.2f}s')
    return df_res


def corpus_eval_faiss(test_data:pd.DataFrame,
                merged_data:pd.DataFrame,
                enc:SentEncode, 
                k:int,
                ind_map:dict,
                vdb:FaissIndex)->pd.DataFrame:
    '''
    Run all queries on index and measure:

        micro-averaged p@i, r@i, f1@i, for 1 =< i =< k
    
    '''
    print('\n==> Evaluation across test set (FAISS)...\n')
    
    begin = time.time()
    df_q = test_data[['query', 'query_id', 'query_docs']].drop_duplicates(keep='first')
    
    print(f'* Shape of test set (queries):\t{df_q.shape}')
    print(f'* Test set snapshot:\n{df_q.sort_values(by="query_docs").head(20)}\n')
    
    results = []
    for _, row in df_q.iterrows():
        _, eval_result = query_eval_faiss(row['query'], 
                                    enc, 
                                    merged_data, 
                                    k, 
                                    ind_map, 
                                    vdb)
        results.append(eval_result)
    df_c = pd.concat(results)
    df_res = df_c.groupby(level=0).mean(numeric_only=True)
    end = time.time()

    print(f'* Time taken: {end-begin:.2f}s')
    return df_res