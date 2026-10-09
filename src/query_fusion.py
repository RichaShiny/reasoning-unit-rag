"""Explicit query retention and fusion for controlled retrieval comparisons."""
import math


def prepare_queries(question, queries, policy='as-provided', baseline=False):
    if policy not in ('as-provided', 'include', 'exclude'):
        raise ValueError('Unknown original-question policy')
    if not isinstance(queries, list) or not queries or any(not isinstance(q, str) or not q.strip() for q in queries):
        raise ValueError('Queries must be a nonempty list of nonempty strings')
    queries = list(dict.fromkeys(q.strip() for q in queries))
    question = question.strip()
    if baseline:
        if queries != [question]:
            raise ValueError('The original condition must contain only the original question')
        return queries
    if policy == 'include':
        return list(dict.fromkeys([question, *queries]))
    if policy == 'exclude':
        queries = [q for q in queries if q != question]
        if not queries:
            raise ValueError('No non-original query remains; select conditions with non-original queries using --query-types')
    return queries


def fuse_scores(matrix, candidate_count, strategy='max', rrf_k=60):
    """Fuse query rankings, sharing competition ranks for exact score ties.

    RRF assigns rank 1, 1, 3 to a tied top pair. An all-tied query therefore
    contributes a constant instead of an arbitrary document-order preference.
    """
    if strategy not in ('max', 'rrf'):
        raise ValueError('Unknown fusion strategy')
    if type(rrf_k) is not int or rrf_k < 1:
        raise ValueError('RRF constant must be a positive integer')
    if not matrix or any(len(row) != candidate_count for row in matrix):
        raise ValueError('Require one score per query and candidate')
    if any(not math.isfinite(score) for row in matrix for score in row):
        raise ValueError('All query scores must be finite')
    if strategy == 'max':
        return [max(row[i] for row in matrix) for i in range(candidate_count)]
    fused = [0.0] * candidate_count
    for row in matrix:
        previous = None
        rank = 0
        for position, i in enumerate(sorted(range(candidate_count), key=lambda i: -row[i]), 1):
            if previous is None or row[i] != previous:
                rank = position
            fused[i] += 1 / (rrf_k + rank)
            previous = row[i]
    return fused
