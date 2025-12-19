import numpy as np
from usearch.index import Index, MetricKind

from ..base.module import BaseANN


class Usearch(BaseANN):
    def __init__(self, metric, method_param):
        self._metric = {
            "angular": MetricKind.Cos,
            "euclidean": MetricKind.L2sq,
        }[metric]
        self._method_param = method_param
        self._ef = 10  # default ef for search

    def fit(self, X):
        self._index = Index(
            ndim=len(X[0]),
            metric=self._metric,
            connectivity=self._method_param["M"],
            expansion_add=self._method_param["efConstruction"],
        )
        # Add all vectors with their indices as keys
        keys = np.arange(len(X))
        self._index.add(keys, np.asarray(X, dtype=np.float32))

    def set_query_arguments(self, ef):
        self._ef = ef
        self.name = "usearch (%s, ef: %d)" % (self._method_param, ef)

    def query(self, v, n):
        results = self._index.search(
            np.asarray(v, dtype=np.float32), 
            count=n, 
            exact=False,
            expansion=self._ef
        )
        return results.keys

    def batch_query(self, X, n):
        results = self._index.search(
            np.asarray(X, dtype=np.float32),
            count=n,
            exact=False,
            expansion=self._ef
        )
        self.res = [r.keys for r in results]

    def get_batch_results(self):
        return self.res

    def freeIndex(self):
        del self._index

    def __str__(self):
        return "usearch (%s, ef: %d)" % (self._method_param, self._ef)

