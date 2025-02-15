# Neural search demos
Neural search demos


```![Search Architecture](./E2E-search.png)```

<p align="center">
  <img src="./E2E-search.png" />
</p>

Cranfield corpus

## Search methods

### Okapi BM25 search

[Okapi BM25](https://en.wikipedia.org/wiki/Okapi_BM25) is a variation of TFIDF.
Consider query $q$ as composed of $n$ keywords $t_i,\dots,t_n$, and assume the target 
document $v$ to match is a bag of $m$ words or terms. Then:

$$
\text{bm25}(q, v) = \sum_{i=1}^n \text{idf}(t_i) * \frac{\text{tf}(t_i, v) * 
        (k_1 + 1)} {\text{tf}(t_i, v) + k_1 * \left(1 - b + b * \frac{m}{avgdl} \right)}
$$

where $\text{tf}(q_{i},v)$ is the number of times that the keyword $t_i$ occurs in $v$, and $avdgl$ is the average document length of a document in the document collection $D$ indexed. $k_1$ and $b$ are hyper-parameters, set by default to $k_1 \in [1.2, 2.0]$ and $b=0.75$. The inverse document frequency of $t_i$ is defined as follows:

$$
\text{idf}(t_i) = \ln \left(\frac{ |D| - n(t_i) + 0.5 } { n(t_i) + 0.5} + 1\right)
$$

with $n(t_i)$ the number of documents in $D$ where $t_i$ appears.

### L2 (Euclidian) search

Consider a $k$-dimensional embedding (vector representation) $\vec{q}$ of query $q$, and 
a $k$-dimensional embedding $\vec{q}$ of document $d$.
Then the [L2-norm]() is defined as the Euclidian distance between $\vec{q}$ and $\vec{d}$, namely:

$$
\text{l2}(q,d) = \sqrt{ \sum_{i=1}^k (\vec{q}_i - \vec{d_i})^2 }
$$

### Reranking with a cross-encoder

###  Reranking with reciprocal rank fusion

$$
  \text{rrf}(d) = \sum_{r \in R} \left( \frac{1}{k + r(d)} \right)
$$

### Evaluation

**Methods**

To compare methods, we rely on IR evaluation metrics, **precision, recall and F1-score at ranking level $k$**. 
What this means is that we:
* restrict $Ans(q)$ to the top $k$ hits $Ans_k^r(q)$ according to ranking metric $r$ (viz., BM25, W2V cosine, bi-encoder cosine)
* check if document $d_i$ at ranking level $1 \leq i \leq k$ is relevant for query $q$

$$
    P{@}k = \frac{TP_k}{TP_k + FP_k} \qquad 
    R{@}k = \frac{TP_k}{TP_k + FN_k} \qquad 
    F1{@}k = \frac{2 \cdot P{@}k \cdot R{@}k}{P{@}k + R{@}k}   
$$

**Results**

<div align="center">

Method                  | P@5      | R@5      | F1@5 
-------                 |------    |------    |------
BM25                    | 0.32     | 0.06     | 0.10
FAISS                   | 0.32     | 0.06     | 0.10
FAISS + cross-encoder   | 0.16     | 0.03     | 0.05
BM25  + cross-encoder   | 0.20     | 0.03     | 0.06
BM25  + FAISS + RRF     | 0.40     | 0.07     | 0.15

</div>

## Running the experiments

```bash
conda create -n search_demo python=3.10.12
conda activate search_demo
pip install -r requirements.txt
python main.py
```
