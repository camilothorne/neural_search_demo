import pandas as pd
import numpy as np
import time

from preprocessing.sent_encoder import SentEncode
from preprocessing.split_data import *

from neural_search.faiss_vdb import FaissIndex
from neural_search.bm25_vdb import BM25Index


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