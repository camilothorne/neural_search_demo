from neural_search import *
from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

from neural_search.sent_encoder import SentEncode
from neural_search.faiss_vdb import FaissIndex
from neural_search.evaluation import EvaluateResults
from neural_search.cross_encoder import ReRanker


def load_cranfield()->pd.DataFrame:
    '''
    Load Cranfield dataset
    '''
    docs = load_dataset('irds/cranfield', 'docs').to_pandas()
    queries = load_dataset('irds/cranfield', 'queries').to_pandas()
    qrels = load_dataset('irds/cranfield', 'qrels').to_pandas()
    queries.rename(columns={'text':'query'}, inplace=True)
    merged_data = queries.merge(qrels, 
        how='inner', on='query_id').merge(docs, how='inner', on='doc_id')
    merged_data['query_docs'] = merged_data[['doc_id', 
        'query_id']].groupby('query_id')['doc_id'].transform('count')
    
    print('\n==> Loading dataset...\n')
    print(list(merged_data.columns))  # Check the merged columns name
    print(merged_data.head())
    print()
    print(merged_data.query_id.value_counts())
    print()
    print(merged_data.doc_id.value_counts())
    print(f'\n* Full data shape: {merged_data.shape}')
    return merged_data


def split_data(df_r:pd.DataFrame, 
               test_queries:int)->tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Split data into train and test
    '''
    print(f'\n==> Splitting into train and test sets...\n')
    query_ids   = Counter(list(df_r.query_id.values))
    most_common = query_ids.most_common(test_queries)
    test        = [id for (id, _) in most_common]
    train_data = df_r[~df_r.query_id.isin(test)]
    test_data  = df_r[df_r.query_id.isin(test)]
    
    print(f'* Number of unique train queries:   {len(query_ids.keys())}')
    print(f'* Number of unique test queries:    {len(test)}')
    print(f'* Number of unique train documents: {len(pd.unique(train_data.doc_id))}')
    print(f'* Number of unique test documents:  {len(pd.unique(test_data.doc_id))}')
    print(f'* Train data shape: {train_data.shape}, test data shape: {test_data.shape}')
    return train_data, test_data


def embedd_and_index_data(df_r:pd.DataFrame, 
                          find:FaissIndex, 
                          enc:SentEncode)->FaissIndex:
    '''
    Update Faiss index
    '''
    print(f'\n==> Embedding and indexing corpus...\n')
    df_r = df_r[['text', 'doc_id']].drop_duplicates(keep='first').reset_index()
    vectors = np.zeros((df_r.shape[0], 384))
    ind_map = {}
    for ind, row in df_r.iterrows():
        vectors[ind,:] = enc.embedd_text(row['text'])
        ind_map[ind] = row['doc_id']
    r, c = vectors.shape
    find.update_index(vectors)
    
    print(f'* Vector data shape: ({r}, {c})')
    print(f'* Index size: {find.index.ntotal} vectors of 384 dimensions')
    return ind_map


def query_eval(test_query:str, 
               enc:SentEncode, 
               merged_data:pd.DataFrame,
               k:int)->pd.DataFrame:
    '''
    Run query on index and measure:

        p@j, r@j, f1@j, for 1 =< j =< k 
    
    '''
    vec_query = np.zeros((1,384))
    vec_query[0,:] = enc.embedd_text(test_query)
    D, I = vdb.search(vec_query, k) # Query index
    df_ans = pd.DataFrame()
    df_ans['scores'] = D[0]
    df_ans['doc_id'] = I[0]
    df_ans['doc_id'] = df_ans['doc_id'].apply(lambda x: ind_map[x])
    ans_docs = list(df_ans['doc_id'].values)
    rel_docs = list(merged_data[
        merged_data['query']==test_query]['doc_id'].values)
    miss_docs = set(rel_docs).difference(set(ans_docs))                       
    df_ans['relevant'] = df_ans['doc_id'].apply(
        lambda x: 'yes' if x in rel_docs else 'no')
    df_ans = df_ans.merge(merged_data[
        ['text', 'doc_id']], on='doc_id', how='inner').drop_duplicates(keep='first')
    eval = EvaluateResults(df_ans, len(miss_docs))
    return eval.compute_performance()


def corpus_eval(test_data:pd.DataFrame, 
               enc:SentEncode, 
               merged_data:pd.DataFrame,
               k:int)->pd.DataFrame:
    '''
    Run all queries on idex and measure:

        micro-averaged p@i, r@i, f1@i, for 1 =< i =< k
    
    '''
    print('\n==> Evaluation across test set...\n')
    df_q = test_data[['query', 'query_id', 'query_docs']].drop_duplicates(keep='first')
    print(f'* Shape of test set (queries):\t{df_q.shape}')
    print(f'* Test set snapshot:\n{df_q.sort_values(by="query_docs").head(20)}')
    results = []
    for _, row in df_q.iterrows():
        eval_result = query_eval(row['query'], enc, merged_data, k)
        results.append(eval_result)
    df_c = pd.concat(results)
    df_res = df_c.groupby(level=0).mean(numeric_only=True)
    return df_res


if __name__ == '__main__':

    num_test_queries = 5
    merged_data = data_cranfield = load_cranfield()
    train_data, test_data = split_data(data_cranfield, test_queries=num_test_queries)
    vdb = FaissIndex(384)
    enc = SentEncode()
    ind_map = embedd_and_index_data(merged_data, vdb, enc)

    print()
    print('==> Querying index...\n')
    
    # Query
    test_query = test_data['query'].head(20).values[10]
    vec_query = np.zeros((1,384))
    vec_query[0,:] = enc.embedd_text(test_query)

    # Answer set
    k = 100 # size of answer set
    D, I = vdb.search(vec_query, k) # Query index
    df_ans = pd.DataFrame()
    df_ans['scores'] = D[0]
    df_ans['doc_id'] = I[0]
    df_ans['doc_id'] = df_ans['doc_id'].apply(lambda x: ind_map[x])
    ans_docs = list(df_ans['doc_id'].values)
    rel_docs = list(merged_data[
        merged_data['query']==test_query]['doc_id'].values)
    miss_docs = set(rel_docs).difference(set(ans_docs))                       
    df_ans['relevant'] = df_ans['doc_id'].apply(
        lambda x: 'yes' if x in rel_docs else 'no')
    df_ans = df_ans.merge(merged_data[
        ['text', 'doc_id']], on='doc_id', how='inner').drop_duplicates(keep='first')
    
    # Visualize
    print(f'* Test query:\t{test_query}')
    print(f'* Size of answer set:\t{k}')
    print(f'* Total relevant documents:\t{len(rel_docs)}')
    print(f'* Missing relevant documents:\t{len(miss_docs)}')
    print(f'* Retrieved relevant documents:\t{df_ans[df_ans["relevant"]=="yes"].shape[0]}')
    print(f'* Shape of answers:\t{df_ans.shape}')
    print(f'* Top answers for test query:\n\n{df_ans.head()}')

    # Evaluation
    eval = EvaluateResults(df_ans, len(miss_docs))
    print()
    print(eval.compute_performance().head(k))

    corpus_eval_result = corpus_eval(test_data, enc, merged_data, k)
    print()
    print(corpus_eval_result.head(k))
    #corpus_eval_result.plot(kind='bar')
    #plt.show()

    # Reranking
    rerank = ReRanker()
    rerank.example()

