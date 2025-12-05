"""
This module supports connecting to a YugabyteDB instance and performing vector
indexing and search using the pgvector extension. YugabyteDB is PostgreSQL-compatible
and supports the pgvector extension.

The following environment variables are used for YugabyteDB connection parameters:

ANN_BENCHMARKS_YUGABYTE_USER      (default: yugabyte)
ANN_BENCHMARKS_YUGABYTE_PASSWORD  (default: yugabyte)
ANN_BENCHMARKS_YUGABYTE_DBNAME    (default: yugabyte)
ANN_BENCHMARKS_YUGABYTE_HOST      (required - no default)
ANN_BENCHMARKS_YUGABYTE_PORT      (default: 5433)
"""

import os
import sys
import threading
import time

import pgvector.psycopg
import psycopg

from typing import Dict, Any, Optional

from ..base.module import BaseANN


METRIC_PROPERTIES = {
    "angular": {
        "distance_operator": "<=>",
        "ops_type": "cosine",
    },
    "euclidean": {
        "distance_operator": "<->",
        "ops_type": "l2",
    }
}


def get_yb_param_env_var_name(param_name: str) -> str:
    return f'ANN_BENCHMARKS_YUGABYTE_{param_name.upper()}'


def get_yb_conn_param(
        param_name: str,
        default_value: Optional[str] = None) -> Optional[str]:
    env_var_name = get_yb_param_env_var_name(param_name)
    env_var_value = os.getenv(env_var_name, default_value)
    if env_var_value is None or len(env_var_value.strip()) == 0:
        return default_value
    return env_var_value


class IndexingProgressMonitor:
    """
    Continuously logs indexing progress, elapsed and estimated remaining
    indexing time.
    """

    MONITORING_DELAY_SEC = 0.5

    def __init__(self, psycopg_connect_kwargs: Dict[str, str]) -> None:
        self.psycopg_connect_kwargs = psycopg_connect_kwargs
        self.monitoring_condition = threading.Condition()
        self.stop_requested = False
        self.psycopg_connect_kwargs = psycopg_connect_kwargs
        self.prev_phase = None
        self.prev_progress_pct = None
        self.prev_tuples_done = None
        self.prev_report_time_sec = None
        self.time_to_load_all_tuples_sec = None
        self._ef_search = None

    def report_progress(
            self,
            phase: str,
            progress_pct: Any,
            tuples_done: Any) -> None:
        if progress_pct is None:
            progress_pct = 0.0
        progress_pct = float(progress_pct)
        if tuples_done is None:
            tuples_done = 0
        tuples_done = int(tuples_done)
        if (phase == self.prev_phase and
                progress_pct == self.prev_progress_pct):
            return
        time_now_sec = time.time()

        elapsed_time_sec = time_now_sec - self.indexing_start_time_sec
        fields = [
            f"Phase: {phase}",
            f"progress: {progress_pct:.1f}%",
            f"elapsed time: {elapsed_time_sec:.3f} sec"
        ]
        if (self.prev_report_time_sec is not None and
            self.prev_tuples_done is not None and
            elapsed_time_sec):
            overall_tuples_per_sec = tuples_done / elapsed_time_sec
            fields.append(
                f"overall tuples/sec: {overall_tuples_per_sec:.2f}")

            time_since_last_report_sec = time_now_sec - self.prev_report_time_sec
            if time_since_last_report_sec > 0:
                cur_tuples_per_sec = ((tuples_done - self.prev_tuples_done) /
                                      time_since_last_report_sec)
                fields.append(
                    f"current tuples/sec: {cur_tuples_per_sec:.2f}")

        remaining_pct = 100 - progress_pct
        if progress_pct > 0 and remaining_pct > 0:
            estimated_remaining_time_sec = \
                elapsed_time_sec / progress_pct * remaining_pct
            estimated_total_time_sec = \
                elapsed_time_sec + estimated_remaining_time_sec
            fields.extend([
                "estimated remaining time: " \
                   f"{estimated_remaining_time_sec:.3f} sec" ,
                f"estimated total time: {estimated_total_time_sec:.3f} sec"
            ])
        print(", ".join(fields))
        sys.stdout.flush()

        self.prev_progress_pct = progress_pct
        self.prev_phase = phase
        self.prev_tuples_done = tuples_done
        self.prev_report_time_sec = time_now_sec

    def monitoring_loop_impl(self, monitoring_cur) -> None:
        while True:
            try:
                monitoring_cur.execute(
                    "SELECT phase, " +
                    "round(100.0 * blocks_done / nullif(blocks_total, 0), 1), " +
                    "tuples_done " +
                    "FROM pg_stat_progress_create_index");
                result_rows = monitoring_cur.fetchall()

                if len(result_rows) == 1:
                    phase, progress_pct, tuples_done = result_rows[0]
                    self.report_progress(phase, progress_pct, tuples_done)
                    if (self.time_to_load_all_tuples_sec is None and
                        phase == 'building index: loading tuples' and
                        progress_pct is not None and
                        float(progress_pct) > 100.0 - 1e-7):
                        self.time_to_load_all_tuples_sec = \
                            time.time() - self.indexing_start_time_sec
                elif len(result_rows) > 0:
                    print(f"Expected exactly one progress result row, got: {result_rows}")
            except Exception as e:
                # YugabyteDB may not support pg_stat_progress_create_index
                print(f"Progress monitoring not available: {e}")
                
            with self.monitoring_condition:
                if self.stop_requested:
                    return
                self.monitoring_condition.wait(
                    timeout=self.MONITORING_DELAY_SEC)
                if self.stop_requested:
                    return

    def monitor_progress(self) -> None:
        try:
            with psycopg.connect(**self.psycopg_connect_kwargs) as monitoring_conn:
                with monitoring_conn.cursor() as monitoring_cur:
                    self.monitoring_loop_impl(monitoring_cur)
        except Exception as e:
            print(f"Progress monitoring connection failed: {e}")

    def start_monitoring_thread(self) -> None:
        self.indexing_start_time_sec = time.time()
        self.monitoring_thread = threading.Thread(target=self.monitor_progress)
        self.monitoring_thread.start()

    def stop_monitoring_thread(self) -> None:
        with self.monitoring_condition:
            self.stop_requested = True
            self.monitoring_condition.notify_all()
        self.monitoring_thread.join()
        self.indexing_time_sec = time.time() - self.indexing_start_time_sec

    def report_timings(self) -> None:
        print(f"YugabyteDB total indexing time: {self.indexing_time_sec:3f} sec")
        if self.time_to_load_all_tuples_sec is not None:
            print("    Time to load all tuples into the index: {:.3f} sec".format(
                self.time_to_load_all_tuples_sec
            ))
            postprocessing_time_sec = \
                self.indexing_time_sec - self.time_to_load_all_tuples_sec
            print("    Index postprocessing time: {:.3f} sec".format(
                postprocessing_time_sec))
        else:
            print("    Detailed breakdown of indexing time not available.")


class YugabyteDB(BaseANN):
    def __init__(self, metric, method_param):
        self._metric = metric
        self._m = method_param['M']
        self._ef_construction = method_param['efConstruction']
        self._cur = None
        self._ef_search = None

        if metric == "angular":
            self._query = "SELECT id FROM items ORDER BY embedding <=> %s LIMIT %s"
        elif metric == "euclidean":
            self._query = "SELECT id FROM items ORDER BY embedding <-> %s LIMIT %s"
        else:
            raise RuntimeError(f"unknown metric {metric}")

    def get_metric_properties(self) -> Dict[str, str]:
        if self._metric not in METRIC_PROPERTIES:
            raise ValueError(
                "Unknown metric: {}. Valid metrics: {}".format(
                    self._metric,
                    ', '.join(sorted(METRIC_PROPERTIES.keys()))
                ))
        return METRIC_PROPERTIES[self._metric]

    def ensure_pgvector_extension_created(self, conn: psycopg.Connection) -> None:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')")
            pgvector_exists = cur.fetchone()[0]
            if pgvector_exists:
                print("vector extension already exists")
            else:
                print("vector extension does not exist, creating")
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

    def fit(self, X):
        psycopg_connect_kwargs: Dict[str, Any] = dict(
            autocommit=True,
        )
        
        # YugabyteDB connection parameters
        psycopg_connect_kwargs['user'] = get_yb_conn_param('user', 'yugabyte')
        psycopg_connect_kwargs['password'] = get_yb_conn_param('password', 'yugabyte')
        psycopg_connect_kwargs['dbname'] = get_yb_conn_param('dbname', 'yugabyte')
        
        # Host is required for YugabyteDB (no default - must be external)
        yb_host = get_yb_conn_param('host')
        if yb_host is None:
            raise ValueError(
                "YugabyteDB host must be specified via ANN_BENCHMARKS_YUGABYTE_HOST environment variable"
            )
        psycopg_connect_kwargs['host'] = yb_host
        
        # Default YugabyteDB YSQL port is 5433
        yb_port_str = get_yb_conn_param('port', '5433')
        psycopg_connect_kwargs['port'] = int(yb_port_str)

        print(f"Connecting to YugabyteDB at {yb_host}:{yb_port_str}...")
        conn = psycopg.connect(**psycopg_connect_kwargs)
        self.ensure_pgvector_extension_created(conn)

        pgvector.psycopg.register_vector(conn)
        cur = conn.cursor()
        cur.execute("DROP TABLE IF EXISTS items")
        cur.execute("CREATE TABLE items (id int, embedding vector(%d))" % X.shape[1])
        
        # Note: STORAGE PLAIN may not be supported in YugabyteDB, wrap in try/except
        try:
            cur.execute("ALTER TABLE items ALTER COLUMN embedding SET STORAGE PLAIN")
        except Exception as e:
            print(f"Note: SET STORAGE PLAIN not supported: {e}")
        
        print("copying data...")
        sys.stdout.flush()
        num_rows = 0
        insert_start_time_sec = time.time()
        
        # YugabyteDB supports COPY command
        with cur.copy("COPY items (id, embedding) FROM STDIN WITH (FORMAT BINARY)") as copy:
            copy.set_types(["int4", "vector"])
            for i, embedding in enumerate(X):
                copy.write_row((i, embedding))
                num_rows += 1
        insert_elapsed_time_sec = time.time() - insert_start_time_sec
        print("inserted {} rows into table in {:.3f} seconds".format(
            num_rows, insert_elapsed_time_sec))

        print("creating index...")
        sys.stdout.flush()
        create_index_str = \
            "CREATE INDEX ON items USING hnsw (embedding vector_%s_ops) " \
            "WITH (m = %d, ef_construction = %d)" % (
                self.get_metric_properties()["ops_type"],
                self._m,
                self._ef_construction
            )
        progress_monitor = IndexingProgressMonitor(psycopg_connect_kwargs)
        progress_monitor.start_monitoring_thread()

        try:
            cur.execute(create_index_str)
        finally:
            progress_monitor.stop_monitoring_thread()
        print("done!")
        progress_monitor.report_timings()
        self._cur = cur

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
            self._cur.execute("SELECT pg_relation_size('items_embedding_idx')")
            return self._cur.fetchone()[0] / 1024
        except Exception:
            # pg_relation_size may behave differently in YugabyteDB
            return 0

    def __str__(self):
        return f"YugabyteDB(m={self._m}, ef_construction={self._ef_construction}, ef_search={self._ef_search})"


