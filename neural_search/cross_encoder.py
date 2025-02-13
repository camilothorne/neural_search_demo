import numpy as np
from sentence_transformers.cross_encoder import CrossEncoder


class ReRanker:

    def __init__(self, path:str="cross-encoder/stsb-distilroberta-base")->None:
        '''
        Instantiate model
        Args:
            path: path to the pre-trained cross-encoder model
        '''
        self.model = CrossEncoder(path)


    def example(self)->None:
        '''
        Example usage:
        Create an instance of ReRanker
        Run example() method to test the reranker
        '''
        # We want to compute the similarity between the query sentence
        query = "A man is eating pasta."
        # With all sentences in the corpus
        corpus = [
            "A man is eating food.",
            "A man is eating a piece of bread.",
            "The girl is carrying a baby.",
            "A man is riding a horse.",
            "A woman is playing violin.",
            "Two men pushed carts through the woods.",
            "A man is riding a white horse on an enclosed ground.",
            "A monkey is playing drums.",
            "A cheetah is running behind its prey.",
        ]
        # Rerank
        self.rerank(query, corpus)
        # Score single sentence
        example = "A cheetah is running behind its prey."
        score = self.score_pair(query, example)
        print(f"\n* Query: {query}, sentence: {example}, score: {score:.5f}")


    def score_pair(self, query:str, sentence:str)->float:
        '''
        Compute similarity score between query and sentence
        Args:
            query: query sentence
            sentence: sentence to be scored
        Returns:
            similarity score
        '''
        return self.model.predict([query, sentence])


    def rerank_corpus(self, query:str, corpus:list[str])->None:
        ''' 
        Rerank answer set based on query
        Args:
            query: query sentence
            corpus: list of sentences to be scored/reranked
        '''

        # 1. We rank all sentences in the corpus for the query
        ranks = self.model.rank(query, corpus)

        # Print the scores
        print("* Query:", query)
        for rank in ranks:
            print(f"\t{rank}")
            print(f"\t{rank['score']:.2f}\t{corpus[rank['corpus_id']]}")

        # 2. Alternatively, you can also manually compute the score between two sentences
        sentence_combinations = [[query, sentence] for sentence in corpus]
        scores = self.model.predict(sentence_combinations)

        # Sort the scores in decreasing order to get the corpus indices
        ranked_indices = np.argsort(scores)[::-1]
        print("* Scores:", scores)
        print("* Indices:", ranked_indices)