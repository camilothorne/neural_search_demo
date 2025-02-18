import pandas as pd
import numpy as np


from preprocessing.sent_encoder import SentEncode
from neural_search.faiss_vdb import FaissIndex
from neural_search.evaluation import EvaluateResults
from neural_search.bm25_vdb import BM25Index


def query_eval_bm25(in_query:str,  
                    merged_data:pd.DataFrame,
                    k:int,
                    ind_map:dict,
                    ddb:BM25Index,
                    print_res:bool=True)->pd.DataFrame:
    '''
    Run query on index and measure:

        p@j, r@j, f1@j, for 1 =< j =< k 
    
    '''
    D, I = ddb.search_index(in_query, k) # query index
    df_ans = pd.DataFrame()
    df_ans['bm25'] = D[0]
    df_ans['doc_id'] = I[0]
    df_ans['doc_id'] = df_ans['doc_id'].apply(lambda x: ind_map[x])
    ans_docs = list(df_ans['doc_id'].values)
    rel_docs = list(merged_data[
        merged_data['query']==in_query]['doc_id'].values)
    miss_docs = set(rel_docs).difference(set(ans_docs))                       
    df_ans['relevant'] = df_ans['doc_id'].apply(
        lambda x: 'yes' if x in rel_docs else 'no')
    df_ans = df_ans.merge(merged_data[
        ['text', 'doc_id']], on='doc_id', how='inner').drop_duplicates(keep='first')
    eval = EvaluateResults(df_ans, len(miss_docs), rank='bm25')
    
    if print_res:
        print(f'* Test query:\t{in_query}')
        print(f'* Size of answer set:\t{k}')
        print(f'* Total relevant documents:\t{len(rel_docs)}')
        print(f'* Missing relevant documents:\t{len(miss_docs)}')
        print(f'* Retrieved relevant documents:\t{df_ans[df_ans["relevant"]=="yes"].shape[0]}')
        print(f'* Shape of answers:\t{df_ans.shape}')
        print(f'* Top answers for test query:\n\n{df_ans.head()}\n')
    
    return df_ans, eval.compute_performance()


def query_eval_faiss(in_query:str, 
               enc:SentEncode, 
               merged_data:pd.DataFrame,
               k:int,
               ind_map:dict,
               vdb:FaissIndex,
               print_res:bool=True)->pd.DataFrame:
    '''
    Run query on index and measure:

        p@j, r@j, f1@j, for 1 =< j =< k 
    
    '''
    vec_query = np.zeros((1,384))
    vec_query[0,:] = enc.embedd_text(in_query)
    D, I = vdb.search(vec_query, k) # query index
    df_ans = pd.DataFrame()
    df_ans['scores'] = D[0]
    df_ans['doc_id'] = I[0]
    df_ans['doc_id'] = df_ans['doc_id'].apply(lambda x: ind_map[x])
    ans_docs = list(df_ans['doc_id'].values)
    rel_docs = list(merged_data[
        merged_data['query']==in_query]['doc_id'].values)
    miss_docs = set(rel_docs).difference(set(ans_docs))                       
    df_ans['relevant'] = df_ans['doc_id'].apply(
        lambda x: 'yes' if x in rel_docs else 'no')
    df_ans = df_ans.merge(merged_data[
        ['text', 'doc_id']], on='doc_id', how='inner').drop_duplicates(keep='first')
    eval = EvaluateResults(df_ans, len(miss_docs))
    
    if print_res:
        print(f'* Test query:\t{in_query}')
        print(f'* Size of answer set:\t{k}')
        print(f'* Total relevant documents:\t{len(rel_docs)}')
        print(f'* Missing relevant documents:\t{len(miss_docs)}')
        print(f'* Retrieved relevant documents:\t{df_ans[df_ans["relevant"]=="yes"].shape[0]}')
        print(f'* Shape of answers:\t{df_ans.shape}')
        print(f'* Top answers for test query:\n\n{df_ans.head()}\n')
    
    return df_ans, eval.compute_performance()