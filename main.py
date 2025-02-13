from neural_search import *
from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import time

from neural_search.sent_encoder import SentEncode
from neural_search.faiss_vdb import FaissIndex
from neural_search.evaluation import EvaluateResults
from neural_search.cross_encoder import ReRanker
from neural_search.bm25_vdb import BM25Index


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
                          enc:SentEncode)->dict:
    '''
    Update Faiss index
    '''
    print(f'\n==> Embedding and indexing corpus (bi-encoder + FAISS + L2 norm)...\n')
    
    begin = time.time()
    df_r = df_r[['text', 'doc_id']].drop_duplicates(keep='first').reset_index()
    vectors = np.zeros((df_r.shape[0], 384))
    ind_map = {}
    for ind, row in df_r.iterrows():
        vectors[ind,:] = enc.embedd_text(row['text'])
        ind_map[ind] = row['doc_id']
    r, c = vectors.shape
    find.update_index(vectors)
    end = time.time()
    
    print(f'* Vector data shape: ({r}, {c})')
    print(f'* Index size: {find.index.ntotal} vectors of 384 dimensions')
    print(f'* Time taken: {end-begin:.2f}s')
    
    return ind_map


def bm25_index_data(df_r:pd.DataFrame, 
                    bm25:BM25Index)->dict:
    '''
    Update BM25 index
    '''
    print(f'\n==> Indexing corpus (BM25)...\n')
    
    begin = time.time()
    df_r = df_r[['text', 'doc_id']].drop_duplicates(keep='first').reset_index()
    corpus = list(df_r.text.values)
    ind_map = {}
    for ind, row in df_r.iterrows():
        ind_map[ind] = row['doc_id']
    bm25.update_index(corpus)
    end = time.time()

    print(f'* Indexed {len(corpus)} documents')
    print(f'* Time taken: {end-begin:.2f}s')

    return ind_map


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
    D, I = ddb.search_index(in_query, k) # Query index
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
    D, I = vdb.search(vec_query, k) # Query index
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


def corpus_eval_faiss(test_data:pd.DataFrame,
                merged_data:pd.DataFrame,
                enc:SentEncode, 
                k:int,
                ind_map:dict,
                vdb:FaissIndex)->pd.DataFrame:
    '''
    Run all queries on idex and measure:

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


def query_rerank(in_query:str, 
                 rank:ReRanker,
                 merged_data:pd.DataFrame,
                 df_ans:pd.DataFrame)->tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Run query on index and measure:

        p@j, r@j, f1@j, for 1 =< j =< k 
    
    '''

    df_ans['rerank'] = df_ans.text.apply(lambda x: rank.score_pair(in_query, x))
    ans_docs = list(df_ans['doc_id'].values)
    rel_docs = list(merged_data[
        merged_data['query']==in_query]['doc_id'].values)
    miss_docs = set(rel_docs).difference(set(ans_docs))                       
    df_ans['relevant'] = df_ans['doc_id'].apply(
        lambda x: 'yes' if x in rel_docs else 'no')
    df_ans = df_ans.merge(merged_data[
        ['text', 'doc_id']], on='doc_id', how='inner').drop_duplicates(keep='first')
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
    Run all queries on idex and measure:

        micro-averaged p@i, r@i, f1@i, for 1 =< i =< k
    
    '''
    print('\n==> Evaluation across test set (rerank)...\n')
    
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


if __name__ == '__main__':

    # ************************************
    # Load dataset and instantiate objects
    # ************************************
    
    num_test_queries = 5 # we use the 5 queries with most relevant documents for the test set
    c_merged_data = data_cranfield = load_cranfield() # load 
    c_train_data, c_test_data = split_data(data_cranfield, test_queries=num_test_queries) # split into training and test sets
    
    v_db  = FaissIndex(384) # FAISS index
    d_db  = BM25Index() # BM25 index
    v_enc = SentEncode() # sentence embedding
    f_rerank = ReRanker() # cross encoder (for FAISS)

    # **********
    # Index data
    # **********

    # Update index (FAISS)
    f_ind_map = embedd_and_index_data(c_merged_data, v_db, v_enc)

    # Update index (BM25)
    d_ind_map = bm25_index_data(c_merged_data, d_db)

    print()
    print('==> Querying index (sample query on: FAISS, FAISS + cross encoder, BM25)...\n')
    
    # Sample query (text)
    c_test_query = c_test_data['query'].head(20).values[10]

    # Answer set size
    k = 100 # size of answer set

    # *********************
    # Test query evaluation
    # *********************

    # Evaluation on test query (FAISS)
    print('a) Sample query on FAISS\n')
    df_ans_q, query_eval_result = query_eval_faiss(c_test_query, 
                                             v_enc, c_merged_data, k, 
                                             f_ind_map, 
                                             v_db)
    print()
    print(query_eval_result.head(k))
    query_eval_result.plot(kind='line', 
                           xlabel='ranking\n' + c_test_query, 
                           ylabel='scores', 
                           title='Base (FAISS)')
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.savefig('results/faiss_base_q.png')

    # Evaluation on test query (FAISS + cross encoder)
    print('\nb) Sample query on FAISS + cross encoder\n')
    df_ans_r, query_rerank_result = query_rerank(c_test_query, 
                                                 f_rerank,
                                                 c_merged_data, 
                                                 df_ans_q)
    print()
    print(query_eval_result.head(k))
    query_rerank_result.plot(kind='line', 
                             xlabel='ranking\n' + c_test_query, 
                             ylabel='scores', 
                             title='Reranked (FAISS + Cross Encoder)')
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.savefig('results/faiss_cross_q.png')

    # Evaluation on test query (BM25)
    print('\nc) Sample query on BM25\n')
    df_ans_q_bm25, query_eval_res_bm25 = query_eval_bm25(c_test_query, 
                                                         c_merged_data, 
                                                         k, f_ind_map, d_db)
    print()
    print(query_eval_res_bm25.head(k))
    query_eval_res_bm25.plot(kind='line', 
                             xlabel='ranking\n' + c_test_query, 
                             ylabel='scores', 
                             title='Base (BM25)')
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.savefig('results/bm25_base_q.png')

    # **********************
    # Evaluation on test set
    # **********************

    # Evaluation across test set (FAISS)
    corpus_eval_result = corpus_eval_faiss(c_test_data,
                                           c_merged_data, v_enc, k, 
                                           f_ind_map, 
                                           v_db)
    print()
    print(corpus_eval_result.head(k))
    corpus_eval_result.plot(kind='line', 
                            xlabel='ranking', 
                            ylabel='scores', 
                            title='Base search - test set (FAISS)')
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.savefig('results/faiss_base_testset.png')

    # Evaluation across test set (BM25)
    corpus_eval_result_bm25 = corpus_eval_bm25(c_test_data, 
                                               c_merged_data,
                                               k, d_ind_map, d_db)
    print()
    print(corpus_eval_result_bm25.head(k))
    corpus_eval_result_bm25.plot(kind='line', 
                                xlabel='ranking', 
                                ylabel='scores', 
                                title='Base search - test set (BM25)')
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.savefig('results/bm25_base_testset.png')

    # Evaluation across test set (FAISS + cross encoder)
    corpus_rerank_result = corpus_rerank_faiss(c_test_data, 
                                               f_rerank,
                                               v_enc, k, f_ind_map, v_db,
                                               c_merged_data)
    print()
    print(corpus_rerank_result.head(k))
    corpus_rerank_result.plot(kind='line',
                              xlabel='ranking', 
                              ylabel='scores', 
                              title='Reranked search - test set (FAISS + Cross Encoder)')
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.savefig('results/faiss_cross_testset.png')