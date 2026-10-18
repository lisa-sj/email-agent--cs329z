"""Problem 3 (bm25): BM25 on an inverted index, from scratch.

Index layout (built in one pass over the corpus):

    postings[term] = (doc_ids: int32 array, tfs: float64 array)   # sorted by doc id
    idf[term]      = Lucene IDF, precomputed
    norm[doc]      = k1 * (1 - b + b * |d| / avgdl), precomputed per document

A search touches only the postings of the query's distinct terms: for each
term it computes the per-document contributions vectorised with NumPy, then
sums contributions per document with np.unique + np.bincount over the
concatenated postings (O(total postings of the query terms), never O(N)).
The index is read-only after build, so it is safe to share across threads.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from cs329z_hw1.tokenizer import tokenize

K1 = 1.5
B = 0.75


class BM25Index:
    def __init__(self, docs: list[str], k1: float = K1, b: float = B) -> None:
        self.k1, self.b = k1, b
        self.n_docs = len(docs)
        lengths = np.zeros(self.n_docs, dtype=np.float64)
        raw: dict[str, tuple[list[int], list[int]]] = {}

        # Single pass: tokenize, count, append to postings.
        for doc_id, text in enumerate(docs):
            tokens = tokenize(text or "")
            lengths[doc_id] = len(tokens)
            for term, tf in Counter(tokens).items():
                entry = raw.get(term)
                if entry is None:
                    raw[term] = ([doc_id], [tf])
                else:
                    entry[0].append(doc_id)
                    entry[1].append(tf)

        avgdl = float(lengths.sum()) / self.n_docs if self.n_docs else 0.0
        self.avgdl = avgdl
        if avgdl > 0:
            self.norm = k1 * (1.0 - b + b * lengths / avgdl)
        else:  # every document is empty: no term can match anyway
            self.norm = np.full(self.n_docs, k1 * (1.0 - b), dtype=np.float64)

        n = self.n_docs
        self.postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.idf: dict[str, float] = {}
        for term, (ids, tfs) in raw.items():
            self.postings[term] = (np.asarray(ids, dtype=np.int64), np.asarray(tfs, dtype=np.float64))
            nt = len(ids)
            self.idf[term] = math.log((n - nt + 0.5) / (nt + 0.5) + 1.0)

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        if k is None or k <= 0:
            return []
        terms = list(dict.fromkeys(tokenize(query or "")))  # distinct, in query order
        terms = [t for t in terms if t in self.postings]
        if not terms:
            return []

        id_parts, score_parts = [], []
        for term in terms:
            ids, tfs = self.postings[term]
            contrib = self.idf[term] * tfs * (self.k1 + 1.0) / (tfs + self.norm[ids])
            id_parts.append(ids)
            score_parts.append(contrib)

        if len(id_parts) == 1:
            cand, scores = id_parts[0], score_parts[0]
        else:
            all_ids = np.concatenate(id_parts)
            cand, inverse = np.unique(all_ids, return_inverse=True)
            scores = np.bincount(inverse, weights=np.concatenate(score_parts), minlength=len(cand))

        # Top-k by (score desc, doc_id asc).
        if len(cand) > k:
            # Keep everything scoring >= the k-th best score so ties at the cutoff survive.
            kth = np.partition(scores, len(scores) - k)[len(scores) - k]
            keep = scores >= kth
            cand, scores = cand[keep], scores[keep]
        order = np.lexsort((cand, -scores))[:k]
        return [(int(cand[i]), float(scores[i])) for i in order]


def build_index(docs: list[str]) -> BM25Index:
    return BM25Index(docs)


def search(index: BM25Index, query: str, k: int) -> list[tuple[int, float]]:
    return index.search(query, k)
