# search_models
Neural search demos

**Okapi BM25 search**

[Okapi BM25](https://en.wikipedia.org/wiki/Okapi_BM25) is a variation of TFIDF.
Consider query $q$ as composed of $n$ keywords $t_i,\dots,t_n$, and assume the target 
document $v$ to match is a bag of $m$ words or terms. Then:

$$
\text{bm25}(q, v) = \sum_{i=1}^n \text{idf}(t_i) * \frac{\text{tf}(t_i, v) * 
        (k_1 + 1)} {\text{tf}(t_i, v) + k_1 * \left(1 - b + b * \frac{m}{avgdl} \right)}
$$

where $\text{tf}(q_{i},v)$ is the number of times that the keyword $t_i$ occurs in $v$, and $avdgl$ is the average document length of a document in the document collection $D$ indexed. $k_1$ and $b$ are hyper-parameters, set by default to $k_1 \in [1.2, 2.0]$ and $b=0.75$.

The inverse document frequency of $t_i$ is defined as follows:

$$
\text{idf}(t_i) = \ln \left(\frac{ |D| - n(t_i) + 0.5 } { n(t_i) + 0.5} + 1\right)
$$

with $n(t_i)$ the number of documents in $D$ where $t_i$ appears.

**L2 (Euclidian) search**

Consider a $k$-dimensional embedding (vector representation) $\vec{q}$ of query $q$, and 
a $k$-dimensional embedding $\vec{q}$ of document $d$.
Then the L2-norm is defined as the Euclidian distance between $\vec{q}$ and $\vec{d}$, namely:

$$
\text{l2}(q,d) = ||\vec{q} - \vec{d}||_2 = \sqrt{ \sum_{i=1}^k (\vec{q}_i - \vec{d_i})^2 }
$$