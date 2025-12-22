#!/bin/bash

echo "Starting YugabyteDB manually..."

# Set paths
YB_HOME="/home/yugabyte"
YSQLSH="$YB_HOME/postgres/bin/ysqlsh"
YB_MASTER="$YB_HOME/bin/yb-master"
YB_TSERVER="$YB_HOME/bin/yb-tserver"

# Create data directories
DATA_DIR="/tmp/yb_data"
mkdir -p $DATA_DIR/master $DATA_DIR/tserver $DATA_DIR/logs
chown -R yugabyte:yugabyte $DATA_DIR

BIND_ADDR="0.0.0.0"

# Check if YugabyteDB is already running by testing YSQL connection
if $YSQLSH -h 127.0.0.1 -p 5433 -U yugabyte -c "SELECT 1;" &>/dev/null; then
    echo "YugabyteDB is already running, skipping startup..."
else
    echo "Starting yb-master..."
    su yugabyte -c "$YB_MASTER \
        --fs_data_dirs=$DATA_DIR/master \
        --rpc_bind_addresses=$BIND_ADDR:7100 \
        --server_broadcast_addresses=127.0.0.1:7100 \
        --master_addresses=127.0.0.1:7100 \
        --replication_factor=1 \
        --yb_num_shards_per_tserver=1 \
        --ysql_num_shards_per_tserver=1 \
        --webserver_port=7000 \
        --webserver_interface=$BIND_ADDR \
        > $DATA_DIR/logs/master.log 2>&1 &"

    echo "Waiting for yb-master to be ready..."
    sleep 15

    echo "Starting yb-tserver..."
    su yugabyte -c "$YB_TSERVER \
        --fs_data_dirs=$DATA_DIR/tserver \
        --rpc_bind_addresses=$BIND_ADDR:9100 \
        --server_broadcast_addresses=127.0.0.1:9100 \
        --tserver_master_addrs=127.0.0.1:7100 \
        --yb_num_shards_per_tserver=1 \
        --ysql_num_shards_per_tserver=1 \
        --start_pgsql_proxy \
        --pgsql_proxy_bind_address=$BIND_ADDR:5433 \
        --webserver_port=9000 \
        --webserver_interface=$BIND_ADDR \
        --cql_proxy_bind_address=$BIND_ADDR:9042 \
        > $DATA_DIR/logs/tserver.log 2>&1 &"
fi

echo "Waiting for YSQL to be ready (this may take 2-3 minutes)..."
MAX_RETRIES=120
for i in $(seq 1 $MAX_RETRIES); do
    if $YSQLSH -h 127.0.0.1 -p 5433 -U yugabyte -c "SELECT 1;" &>/dev/null; then
        echo "YSQL is ready!"
        break
    fi
    if [ $i -eq $MAX_RETRIES ]; then
        echo "ERROR: YSQL failed to start within the expected time"
        echo "=== Master log ==="
        tail -50 $DATA_DIR/logs/master.log 2>/dev/null || true
        echo "=== TServer log ==="
        tail -50 $DATA_DIR/logs/tserver.log 2>/dev/null || true
        exit 1
    fi
    if [ $((i % 10)) -eq 0 ]; then
        echo "Waiting for YSQL... ($i/$MAX_RETRIES)"
    fi
    sleep 2
done

echo "Configuring database..."
# Create user if not exists
$YSQLSH -h 127.0.0.1 -p 5433 -U yugabyte -c "CREATE USER ann WITH PASSWORD 'ann';" 2>/dev/null || echo "User ann might already exist"

# Create database - retry a few times
DB_CREATED=false
for i in $(seq 1 10); do
    if $YSQLSH -h 127.0.0.1 -p 5433 -U yugabyte -c "CREATE DATABASE ann;" 2>/dev/null; then
        echo "Database ann created successfully"
        DB_CREATED=true
        break
    fi
    # Check if database already exists
    if $YSQLSH -h 127.0.0.1 -p 5433 -U yugabyte -d ann -c "SELECT 1;" 2>/dev/null; then
        echo "Database ann already exists"
        DB_CREATED=true
        break
    fi
    echo "Waiting for database creation... ($i/10)"
    sleep 3
done

if [ "$DB_CREATED" = false ]; then
    echo "ERROR: Failed to create or verify database ann"
    exit 1
fi

$YSQLSH -h 127.0.0.1 -p 5433 -U yugabyte -c "GRANT ALL PRIVILEGES ON DATABASE ann TO ann;" 2>/dev/null || true

echo "Creating extension in ann database..."
# Retry extension creation as it may conflict with concurrent processes
for i in $(seq 1 5); do
    if $YSQLSH -h 127.0.0.1 -p 5433 -U yugabyte -d ann -c "CREATE EXTENSION IF NOT EXISTS vector;" 2>/dev/null; then
        echo "Extension vector created successfully"
        break
    fi
    echo "Retrying extension creation... ($i/5)"
    sleep 2
done

$YSQLSH -h 127.0.0.1 -p 5433 -U yugabyte -d ann -c "GRANT ALL ON SCHEMA public TO ann;" 2>/dev/null || true

# Tuning
$YSQLSH -h 127.0.0.1 -p 5433 -U yugabyte -c "ALTER USER ann SET maintenance_work_mem = '4GB';" 2>/dev/null || true
$YSQLSH -h 127.0.0.1 -p 5433 -U yugabyte -c "ALTER USER ann SET max_parallel_maintenance_workers = 0;" 2>/dev/null || true

echo "Current processes:"
ps aux | grep -E "(yb-|postgres)" | grep -v grep || true

echo "Starting benchmark..."
python3 -u run_algorithm.py "$@"
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "Benchmark failed with exit code $EXIT_CODE"
    echo "=== Master log ==="
    tail -100 $DATA_DIR/logs/master.log 2>/dev/null || true
    echo "=== TServer log ==="
    tail -100 $DATA_DIR/logs/tserver.log 2>/dev/null || true
fi

exit $EXIT_CODE
