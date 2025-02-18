from neural_search import *
import pandas as pd
import time


from preprocessing.split_data import *
from neural_search.query_eval import *
from neural_search.corpus_eval import *


from preprocessing.sent_encoder import SentEncode
from neural_search.faiss_vdb import FaissIndex
from neural_search.evaluation import EvaluateResults
from neural_search.cross_encoder import ReRanker
from neural_search.bm25_vdb import BM25Index
from neural_search.rrf import RRF



def query_rerank(in_query:str, 
                 rank:ReRanker,
                 merged_data:pd.DataFrame,
                 df_ans:pd.DataFrame,
                 rrf:bool=False)->tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Run query on index and measure:

        p@j, r@j, f1@j, for 1 =< j =< k 
    
    '''
    if rrf==False:
        # RRF will generate this score
        df_ans['rerank'] = df_ans.text.apply(lambda x: rank.score_pair(in_query, x))
    ans_docs = list(df_ans['doc_id'].values)
    rel_docs = list(merged_data[
        merged_data['query']==in_query]['doc_id'].values)
    miss_docs = set(rel_docs).difference(set(ans_docs)) 

    if 'relevant' not in list(df_ans.columns):
        df_ans['relevant'] = df_ans['doc_id'].apply(
           lambda x: 'yes' if x in rel_docs else 'no')
    
    if 'text' not in list(df_ans.columns):
        df_ans = df_ans.merge(merged_data[
           ['text', 'doc_id']], on='doc_id', how='inner').drop_duplicates(keep='first')
    
    # The 'rerank' option will sort by score in descending order 
    # (from highest to lowest) as scores are now either RRF or
    # similarity scores 
    eval = EvaluateResults(df_ans, len(miss_docs), rank='rerank')
    return df_ans, eval.compute_performance()


def corpus_rerank_faiss(test_data:pd.DataFrame, 
                    rank:ReRanker,
                    enc:SentEncode, 
                    k:int,
                    ind_map:dict,
                    vdb:FaissIndex,  # FAISS index
                    merged_data:pd.DataFrame,  # merged dataset with doc_id and text
                   )->pd.DataFrame:
    '''
    Run all queries on index and measure:

        micro-averaged p@i, r@i, f1@i, for 1 =< i =< k
    
    '''
    print('\n==> Evaluation across test set (FAISS + Cross Encoder)...\n')
    
    begin = time.time()
    df_q = test_data[['query', 'query_id', 'query_docs']].drop_duplicates(keep='first')
    
    print(f'* Shape of test set (queries):\t{df_q.shape}')
    print(f'* Test set snapshot:\n{df_q.sort_values(by="query_docs").head(20)}\n')
    
    results = []
    for _, row in df_q.iterrows():
        df_ans, _ = query_eval_faiss(row['query'], 
                                    enc, 
                                    merged_data, 
                                    k, 
                                    ind_map, 
                                    vdb, 
                                    print_res=False)
        _, eval_result = query_rerank(row['query'], rank, merged_data, df_ans)
        results.append(eval_result)
    df_c = pd.concat(results)
    df_res = df_c.groupby(level=0).mean(numeric_only=True)
    end = time.time()
    print(f'* Time taken: {end-begin:.2f}s')
    return df_res


def corpus_rerank_bm25(test_data:pd.DataFrame, 
                    rank:ReRanker,
                    k:int,
                    ind_map:dict,
                    ddb:BM25Index,  # index
                    merged_data:pd.DataFrame,  # merged dataset with doc_id and text
                   )->pd.DataFrame:
    '''
    Run all queries on index and measure:

        micro-averaged p@i, r@i, f1@i, for 1 =< i =< k
    
    '''
    print('\n==> Evaluation across test set (BM25 + Cross Encoder)...\n')
    
    begin = time.time()
    df_q = test_data[['query', 'query_id', 'query_docs']].drop_duplicates(keep='first')
    
    print(f'* Shape of test set (queries):\t{df_q.shape}')
    print(f'* Test set snapshot:\n{df_q.sort_values(by="query_docs").head(20)}\n')
    
    results = []
    for _, row in df_q.iterrows():
        df_ans, _ = query_eval_bm25(row['query'], 
                                    merged_data, 
                                    k, 
                                    ind_map, 
                                    ddb, 
                                    print_res=False)
        _, eval_result = query_rerank(row['query'], rank, merged_data, df_ans)
        results.append(eval_result)
    df_c = pd.concat(results)
    df_res = df_c.groupby(level=0).mean(numeric_only=True)
    end = time.time()

    print(f'* Time taken: {end-begin:.2f}s')
    return df_res


def corpus_rerank_rrf(test_data:pd.DataFrame, 
                    rank:ReRanker,
                    k:int,
                    enc:SentEncode,
                    dind_map:dict,
                    vind_map:dict,
                    ddb:BM25Index,  # index1
                    vdb:FaissIndex, # index2
                    merged_data:pd.DataFrame,  # merged dataset with doc_id and text
                   )->pd.DataFrame:
    '''
    Run all queries on index and measure:

        micro-averaged p@i, r@i, f1@i, for 1 =< i =< k
    
    '''
    print('\n==> Evaluation across test set (BM25 + FAISS + RRF)...\n')
    
    begin = time.time()
    df_q = test_data[['query', 'query_id', 'query_docs']].drop_duplicates(keep='first')
    
    print(f'* Shape of test set (queries):\t{df_q.shape}')
    print(f'* Test set snapshot:\n{df_q.sort_values(by="query_docs").head(20)}\n')
    
    results = []
    for _, row in df_q.iterrows():
        df_ans1, _ = query_eval_bm25(row['query'], 
                                    merged_data, 
                                    k, 
                                    dind_map, 
                                    ddb, 
                                    print_res=False)
        df_ans2, _ = query_eval_faiss(row['query'], 
                                    enc, 
                                    merged_data, 
                                    k, 
                                    vind_map, 
                                    vdb, 
                                    print_res=False)
        rrf = RRF(df_ans1, df_ans2)
        df_ans = rrf.get_rrf()
        _, eval_result = query_rerank(row['query'], rank, merged_data, df_ans, rrf=True)
        results.append(eval_result)
    df_c = pd.concat(results)
    df_res = df_c.groupby(level=0).mean(numeric_only=True)
    end = time.time()

    print(f'* Time taken: {end-begin:.2f}s')
    return df_res