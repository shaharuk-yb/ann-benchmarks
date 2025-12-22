import os
import sys
import time
import pgvector.psycopg
import psycopg
from ..base.module import BaseANN

METRIC_PROPERTIES = {
    "angular": {
        "distance_operator": "<=>",
        "ops_type": "vector_cosine_ops",
    },
    "euclidean": {
        "distance_operator": "<->",
        "ops_type": "vector_l2_ops",
    }
}

class YugabyteDB(BaseANN):
    def __init__(self, metric, method_param):
        self._metric = metric
        self._m = method_param['M']
        self._ef_construction = method_param['efConstruction']
        self._cur = None
        self._ef_search = None

        if metric not in METRIC_PROPERTIES:
             raise RuntimeError(f"unknown metric {metric}")
        
        props = METRIC_PROPERTIES[metric]
        self._ops_type = props["ops_type"]
        self._query = f"SELECT id FROM items ORDER BY embedding {props['distance_operator']} %s LIMIT %s"

    def fit(self, X):
        # Get port from environment variable (set by entrypoint.sh for parallelism support)
        ysql_port = int(os.environ.get("YSQL_PORT", "5433"))
        
        conn_args = {
            "host": "127.0.0.1",
            "port": ysql_port,
            "user": "ann",
            "password": "ann", 
            "dbname": "ann",
            "autocommit": True
        }
        
        print("Connecting to YugabyteDB...")
        for i in range(30): # Wait up to 60s
            try:
                conn = psycopg.connect(**conn_args)
                break
            except Exception as e:
                if i % 5 == 0:
                    print(f"Connection attempt {i} failed: {e}")
                time.sleep(2)
        else:
             raise RuntimeError("Could not connect to YugabyteDB")

        # Retry logic for CREATE EXTENSION to handle concurrent access with parallelism
        max_extension_retries = 10
        for attempt in range(max_extension_retries):
            try:
                with conn.cursor() as cur:
                    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                break
            except psycopg.errors.SerializationFailure as e:
                if attempt < max_extension_retries - 1:
                    print(f"Extension creation conflict, retrying ({attempt + 1}/{max_extension_retries})...")
                    time.sleep(1 + attempt * 0.5)  # Backoff
                    continue
                raise
            except Exception as e:
                # Extension might already exist, which is fine
                if "already exists" in str(e).lower():
                    break
                raise

        pgvector.psycopg.register_vector(conn)
        cur = conn.cursor()
        
        cur.execute("DROP TABLE IF EXISTS items")
        cur.execute("CREATE TABLE items (id int, embedding vector(%d))" % X.shape[1])
        
        # STORAGE PLAIN optimization
        try:
            cur.execute("ALTER TABLE items ALTER COLUMN embedding SET STORAGE PLAIN")
        except Exception as e:
            print(f"Warning: Could not set storage plain (might be unsupported): {e}")

        print("Copying data...")
        sys.stdout.flush()
        num_rows = 0
        insert_start_time_sec = time.time()

        with cur.copy("COPY items (id, embedding) FROM STDIN WITH (FORMAT BINARY)") as copy:
            copy.set_types(["int4", "vector"])
            for i, embedding in enumerate(X):
                copy.write_row((i, embedding))
                num_rows += 1
        
        insert_elapsed_time_sec = time.time() - insert_start_time_sec
        print("inserted {} rows into table in {:.3f} seconds".format(
            num_rows, insert_elapsed_time_sec))

        print("Creating index...")
        sys.stdout.flush()
        create_index_str = \
            "CREATE INDEX items_embedding_idx ON items USING hnsw (embedding %s) " \
            "WITH (m = %d, ef_construction = %d)" % (
                self._ops_type,
                self._m,
                self._ef_construction
            )

        # Retry logic for YugabyteDB catalog version mismatch errors
        start = time.time()
        max_retries = 5
        for attempt in range(max_retries):
            try:
                cur.execute(create_index_str)
                print(f"Index created in {time.time() - start} seconds")
                break
            except Exception as e:
                if "Catalog Version Mismatch" in str(e) and attempt < max_retries - 1:
                    print(f"Catalog version mismatch, retrying ({attempt + 1}/{max_retries})...")
                    time.sleep(2)
                    continue
                raise
        
        self._cur = cur
        self._conn = conn

    def set_query_arguments(self, ef_search):
        self._ef_search = ef_search
        self._cur.execute("SET hnsw.ef_search = %d" % ef_search)

    def query(self, v, n):
        self._cur.execute(self._query, (v, n), binary=True, prepare=True)
        return [id for id, in self._cur.fetchall()]

    def get_memory_usage(self):
        if self._cur is None:
            return 0
        try:
            # YugabyteDB uses pg_table_size() for index sizes (pg_relation_size returns 0)
            self._cur.execute("SELECT pg_table_size('items_embedding_idx')")
            return self._cur.fetchone()[0] / 1024
        except Exception:
            return 0

    def __str__(self):
        return f"YugabyteDB(m={self._m}, ef_construction={self._ef_construction}, ef_search={self._ef_search})"
