import numpy as np
from usearch.index import Index, MetricKind, ScalarKind

from ..base.module import BaseANN


class Usearch(BaseANN):
    # Map string dtype names to ScalarKind and numpy dtypes
    DTYPE_MAP = {
        "f32": (ScalarKind.F32, np.float32),
        "f16": (ScalarKind.F16, np.float16),
        "bf16": (ScalarKind.BF16, np.float16),  # numpy doesn't have bf16, use f16 for input
        "i8": (ScalarKind.I8, np.float32),  # i8 quantization happens internally
    }

    def __init__(self, metric, method_param):
        self._metric = {
            "angular": MetricKind.Cos,
            "euclidean": MetricKind.L2sq,
        }[metric]
        self._method_param = method_param
        self._ef = 10  # default ef for search
        # Default to f16 as recommended by usearch author
        self._dtype_name = method_param.get("dtype", "f16")
        self._scalar_kind, self._np_dtype = self.DTYPE_MAP.get(
            self._dtype_name, (ScalarKind.F16, np.float16)
        )

    def fit(self, X):
        self._index = Index(
            ndim=len(X[0]),
            metric=self._metric,
            dtype=self._scalar_kind,
            connectivity=self._method_param["M"],
            expansion_add=self._method_param["efConstruction"],
            expansion_search=self._ef,
        )
        # Add all vectors with their indices as keys
        keys = np.arange(len(X))
        self._index.add(keys, np.asarray(X, dtype=self._np_dtype))
        
        # Log SIMD/ISA information for verification
        isa_name = self._index.specs.isa_name if hasattr(self._index.specs, 'isa_name') else "unknown"
        print(f"USearch index created: dtype={self._dtype_name}, ISA={isa_name}")

    def set_query_arguments(self, ef):
        self._ef = ef
        # Set the expansion factor for search on the index
        self._index.expansion_search = ef
        self.name = f"usearch ({self._method_param}, ef: {ef}, dtype: {self._dtype_name})"

    def query(self, v, n):
        results = self._index.search(
            np.asarray(v, dtype=self._np_dtype), 
            count=n, 
            exact=False
        )
        return results.keys

    def batch_query(self, X, n):
        results = self._index.search(
            np.asarray(X, dtype=self._np_dtype),
            count=n,
            exact=False
        )
        self.res = [r.keys for r in results]

    def get_batch_results(self):
        return self.res

    def freeIndex(self):
        del self._index

    def __str__(self):
        return f"usearch ({self._method_param}, ef: {self._ef}, dtype: {self._dtype_name})"
