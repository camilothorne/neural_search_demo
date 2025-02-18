import numpy as np
from sentence_transformers.cross_encoder import CrossEncoder


class ReRanker:

    def __init__(self, path:str="cross-encoder/stsb-distilroberta-base", out_dim:int=1)->None:
        '''
        Instantiate model
        Args:
            path: path to the pre-trained cross-encoder model
            out_dim: dimension of the output vector (1 or 2, with the last referring to 'relevant')
        '''
        self.model = CrossEncoder(path)
        self.out_dim = out_dim


    def example(self)->None:
        '''
        Example usage:
        Create an instance of ReRanker
        Run example() method to test the reranker
        '''
        # We want to compute the similarity between the query sentence
        query = "A man is eating pasta."

        # Score sentences
        example1 = "A cheetah is running behind its prey."
        score = self.score_pair(query, example1)
        print(f"\n* Query: {query}, sentence: {example1}, score: {score:.5f}")
        example2 = "A man is eating a pizza."
        score = self.score_pair(query, example2)
        print(f"* Query: {query}, sentence: {example2}, score: {score:.5f}")
        example3 = "A man is riding a white horse on an enclosed ground."
        score = self.score_pair(query, example3)
        print(f"* Query: {query}, sentence: {example3}, score: {score:.5f}")
        example4 = "A man is eating a piece of bread."
        score = self.score_pair(query, example4)
        print(f"* Query: {query}, sentence: {example4}, score: {score:.5f}")


    def score_pair(self, query:str, sentence:str)->float:
        '''
        Compute relevance score between query and sentence
        Args:
            query: query sentence
            sentence: sentence to be scored
        Returns:
            reranking score
        '''
        score = self.model.predict([query, sentence], convert_to_numpy=True, 
                                  convert_to_tensor=False, apply_softmax=True)
        if self.out_dim == 1:
            return score # result is a scalar!
        else:
            return score[0] # result is a pair, so return score for relevant class only


    def rerank_corpus(self, query:str, corpus:list[str])->tuple[list,list]:
        ''' 
        Rerank answer set based on query
        Args:
            query: query sentence
            corpus: list of sentences to be scored/reranked
        '''

        print("* Query:", query)

        # Manually compute the score between two sentences
        sentence_combinations = [[query, sentence] for sentence in corpus]
        scores = self.model.predict(sentence_combinations)

        # Sort the scores in decreasing order to get the corpus indices
        ranked_indices = np.argsort(scores)[::-1]
        
        print("* Scores:", scores)
        print("* Indices:", ranked_indices)

        return ranked_indices, scores