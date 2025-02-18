import matplotlib.pyplot as plt


from preprocessing.split_data import *
from neural_search.reranking import *
from neural_search.query_eval import *
from neural_search.corpus_eval import *
from neural_search.indexing import *


from preprocessing.sent_encoder import SentEncode
from neural_search.faiss_vdb import FaissIndex
from neural_search.cross_encoder import ReRanker
from neural_search.bm25_vdb import BM25Index
from neural_search.rrf import RRF


if __name__ == '__main__':

    # ************************************
    # Load dataset and instantiate objects
    # ************************************
    
    num_test_queries = 5 # we use the 5 queries with most relevant documents for the test set
    c_merged_data = data_cranfield = load_cranfield() # load 
    c_train_data, c_test_data = split_data(data_cranfield, 
                                           test_queries=num_test_queries) # split into training and test sets
    
    v_db  = FaissIndex(384) # FAISS index
    d_db  = BM25Index() # BM25 index
    v_enc = SentEncode() # sentence embedding
    #f_rerank = ReRanker() # default cross encoder (for FAISS)
    f_rerank = ReRanker(path="./models/training_cranfield-2025-02-17_18-47-23", out_dim=2) # custom cross encoder 

    f_rerank.example()

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
    # Save one example (debug)
    df_ans_q.to_csv('results/' + c_test_query  + '_faiss.tsv', 
                    sep='\t', index=False, escapechar='\\') 
    df_ans_q.to_json('results/' + c_test_query + '_faiss.json', 
                     index=False, orient='records', lines=True)
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
    df_r, query_rerank_result = query_rerank(c_test_query, 
                                                 f_rerank,
                                                 c_merged_data, 
                                                 df_ans_q)
   # Save one example (debug)
    df_r[['scores','rerank', 'relevant']].to_csv('results/' + c_test_query  + '_faiss_cross.tsv', 
                                                 sep='\t', index=False, escapechar='\\')
    df_r.to_json('results/' + c_test_query + '_faiss_cross.json', 
                 index=False, orient='records', lines=True) 
    print()
    print(query_rerank_result.head(k))
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

    # Evaluation on test query (BM25 + cross encoder)
    print('\nd) Sample query on BM25 + cross encoder\n')
    _, query_rerank_result_bm25 = query_rerank(c_test_query, 
                                                 f_rerank,
                                                 c_merged_data, 
                                                 df_ans_q_bm25)
    print()
    print(query_rerank_result_bm25.head(k))
    query_rerank_result_bm25.plot(kind='line', 
                             xlabel='ranking\n' + c_test_query, 
                             ylabel='scores', 
                             title='Reranked (FAISS + Cross Encoder)')
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.savefig('results/bm25_cross_q.png')


    # Evaluation on test query (BM25 + FAISS + RRF)
    print('\ne) Sample query on BM25 + FAISS + RRF\n')
    q_rrf = RRF(df_ans_q_bm25, df_ans_q) # merge results and rerank
    df_ans_q_rrf = q_rrf.get_rrf()
    _, query_rerank_result_rrf = query_rerank(c_test_query, 
                                                 f_rerank,
                                                 c_merged_data, 
                                                 df_ans_q_rrf,
                                                 rrf=True)
    print()
    print(query_rerank_result_rrf.head(k))
    query_rerank_result_rrf.plot(kind='line', 
                             xlabel='ranking\n' + c_test_query, 
                             ylabel='scores', 
                             title='Reranked (FAISS + BM25 + RRF)')
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.savefig('results/bm25_faiss_rrf_q.png')

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


    # Evaluation across test set (BM25 + cross encoder)
    corpus_rerank_res_bm25 = corpus_rerank_bm25(c_test_data, 
                                                f_rerank,
                                                k, d_ind_map, d_db,
                                                c_merged_data)
    print()
    print(corpus_rerank_res_bm25.head(k))
    corpus_rerank_res_bm25.plot(kind='line',
                              xlabel='ranking', 
                              ylabel='scores', 
                              title='Reranked search - test set (BM25 + Cross Encoder)')
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.savefig('results/bm25_cross_testset.png')


    # Evaluation across test set (BM25 + FAISS + RRF)
    corpus_rerank_res_rrf = corpus_rerank_rrf(c_test_data, 
                                              f_rerank,
                                              k,
                                              v_enc,
                                              d_ind_map,
                                              f_ind_map,
                                              d_db,  # index1
                                              v_db, # index2
                                              c_merged_data)
    print()
    print(corpus_rerank_res_rrf.head(k))
    corpus_rerank_res_rrf.plot(kind='line',
                              xlabel='ranking', 
                              ylabel='scores', 
                              title='Reranked search - test set (BM25 + FAISS + RRF)')
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.savefig('results/bm25_faiss_rrf_testset.png')