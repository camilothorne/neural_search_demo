from datasets import load_dataset
import pandas as pd
from collections import Counter


def load_cranfield()->pd.DataFrame:
    '''
    Load Cranfield dataset

    Returns:
        pd.DataFrame: merged dataset with query, query_id, doc_id, and query_docs columns
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
    print(list(merged_data.columns))  # check the merged columns name
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

    Args:
        df_r: dataset
        test_queries: number of queries for the test set
    Returns:
        tuple: train_data, test_data
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


def negative_mining(df_r:pd.DataFrame)->pd.DataFrame:
    '''
    Negative mining to create negative relevance examples

    Args:
        df_r: training dataset
    Returns:
        DataFrame with negative relevance examples for training set
    '''
    print('\n==> Negative mining for training set...\n')
    
    queries = set(list(df_r.query_id.values))
    res_data = []
    for q_id in queries:
        pos_df = df_r[df_r.query_id==q_id][['query', 'text', 'query_id', 'doc_id']]
        pos_q  = list(pos_df['query'].values)[0]
        neg_df = df_r[~df_r.query_id.isin([q_id])][['text','doc_id']].sample(pos_df.shape[0], random_state=42)
        pos_df['relevant'] = 'yes'
        neg_df['relevant'] = 'no'
        neg_df['query']    = pos_q
        neg_df['query_id'] = q_id
        res_data.append(pos_df)
        res_data.append(neg_df)
    res_df = pd.concat(res_data)

    print(f'* Original shape: {df_r.shape}')
    print(f'* Negative mining result shape: {res_df.shape}')
    print(f'* Sample positives:\n{res_df[res_df.relevant=="yes"].head()}')
    print(f'\n* Sample negatives:\n{res_df[res_df.relevant=="no"].head()}')
    print(f'* Distribution:\n{res_df.relevant.value_counts()}')
    
    return res_df
