#!/bin/bash

echo "Starting YugabyteDB manually..."

# Set paths
YB_HOME="/home/yugabyte"
YSQLSH="$YB_HOME/postgres/bin/ysqlsh"
YB_MASTER="$YB_HOME/bin/yb-master"
YB_TSERVER="$YB_HOME/bin/yb-tserver"

# Use dynamic ports and directories based on a hash of the container ID to avoid conflicts
# when running with --parallelism > 1 and network_mode="host"
CONTAINER_ID=$(cat /proc/self/cgroup 2>/dev/null | grep -o '[0-9a-f]\{12,\}' | head -1 || echo "$$")

# Create unique data directory per container
DATA_DIR="/tmp/yb_data_${CONTAINER_ID:0:12}"
mkdir -p $DATA_DIR/master $DATA_DIR/tserver $DATA_DIR/logs
chown -R yugabyte:yugabyte $DATA_DIR
# Generate a port offset (0-999) from container ID to spread ports across range
PORT_OFFSET=$(echo "$CONTAINER_ID" | cksum | awk '{print $1 % 1000}')

# Master ports
MASTER_RPC_PORT=$((7100 + PORT_OFFSET))
MASTER_WEB_PORT=$((7000 + PORT_OFFSET))

# TServer ports
TSERVER_RPC_PORT=$((9100 + PORT_OFFSET))
TSERVER_WEB_PORT=$((9000 + PORT_OFFSET))

# YSQL (PostgreSQL) ports
YSQL_PORT=$((5433 + PORT_OFFSET))
YSQL_WEB_PORT=$((13000 + PORT_OFFSET))

# CQL (Cassandra) ports - even if not used, must be unique
CQL_PORT=$((9042 + PORT_OFFSET))
CQL_WEB_PORT=$((12000 + PORT_OFFSET))

echo "Using ports: YSQL=$YSQL_PORT, Master RPC=$MASTER_RPC_PORT, TServer RPC=$TSERVER_RPC_PORT"

# Export YSQL_PORT so module.py can use it
export YSQL_PORT

BIND_ADDR="0.0.0.0"

# Check if YugabyteDB is already running on our assigned port
if $YSQLSH -h 127.0.0.1 -p $YSQL_PORT -U yugabyte -c "SELECT 1;" &>/dev/null; then
    echo "YugabyteDB is already running on port $YSQL_PORT, skipping startup..."
else
    echo "Starting yb-master on port $MASTER_RPC_PORT..."
    su yugabyte -c "$YB_MASTER \
        --fs_data_dirs=$DATA_DIR/master \
        --rpc_bind_addresses=$BIND_ADDR:$MASTER_RPC_PORT \
        --server_broadcast_addresses=127.0.0.1:$MASTER_RPC_PORT \
        --master_addresses=127.0.0.1:$MASTER_RPC_PORT \
        --replication_factor=1 \
        --yb_num_shards_per_tserver=1 \
        --ysql_num_shards_per_tserver=1 \
        --webserver_port=$MASTER_WEB_PORT \
        --webserver_interface=$BIND_ADDR \
        > $DATA_DIR/logs/master.log 2>&1 &"

    echo "Waiting for yb-master to be ready..."
    sleep 15

    echo "Starting yb-tserver on YSQL port $YSQL_PORT..."
    su yugabyte -c "$YB_TSERVER \
        --fs_data_dirs=$DATA_DIR/tserver \
        --rpc_bind_addresses=$BIND_ADDR:$TSERVER_RPC_PORT \
        --server_broadcast_addresses=127.0.0.1:$TSERVER_RPC_PORT \
        --tserver_master_addrs=127.0.0.1:$MASTER_RPC_PORT \
        --yb_num_shards_per_tserver=1 \
        --ysql_num_shards_per_tserver=1 \
        --start_pgsql_proxy \
        --pgsql_proxy_bind_address=$BIND_ADDR:$YSQL_PORT \
        --pgsql_proxy_webserver_port=$YSQL_WEB_PORT \
        --webserver_port=$TSERVER_WEB_PORT \
        --webserver_interface=$BIND_ADDR \
        --cql_proxy_bind_address=$BIND_ADDR:$CQL_PORT \
        --cql_proxy_webserver_port=$CQL_WEB_PORT \
        > $DATA_DIR/logs/tserver.log 2>&1 &"
fi

echo "Waiting for YSQL to be ready on port $YSQL_PORT (this may take 2-3 minutes)..."
MAX_RETRIES=120
for i in $(seq 1 $MAX_RETRIES); do
    if $YSQLSH -h 127.0.0.1 -p $YSQL_PORT -U yugabyte -c "SELECT 1;" &>/dev/null; then
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
$YSQLSH -h 127.0.0.1 -p $YSQL_PORT -U yugabyte -c "CREATE USER ann WITH PASSWORD 'ann';" 2>/dev/null || echo "User ann might already exist"

# Create database - retry a few times
DB_CREATED=false
for i in $(seq 1 10); do
    if $YSQLSH -h 127.0.0.1 -p $YSQL_PORT -U yugabyte -c "CREATE DATABASE ann;" 2>/dev/null; then
        echo "Database ann created successfully"
        DB_CREATED=true
        break
    fi
    # Check if database already exists
    if $YSQLSH -h 127.0.0.1 -p $YSQL_PORT -U yugabyte -d ann -c "SELECT 1;" 2>/dev/null; then
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

$YSQLSH -h 127.0.0.1 -p $YSQL_PORT -U yugabyte -c "GRANT ALL PRIVILEGES ON DATABASE ann TO ann;" 2>/dev/null || true

echo "Creating extension in ann database..."
# Retry extension creation as it may conflict with concurrent processes
for i in $(seq 1 5); do
    if $YSQLSH -h 127.0.0.1 -p $YSQL_PORT -U yugabyte -d ann -c "CREATE EXTENSION IF NOT EXISTS vector;" 2>/dev/null; then
        echo "Extension vector created successfully"
        break
    fi
    echo "Retrying extension creation... ($i/5)"
    sleep 2
done

$YSQLSH -h 127.0.0.1 -p $YSQL_PORT -U yugabyte -d ann -c "GRANT ALL ON SCHEMA public TO ann;" 2>/dev/null || true

# Tuning
$YSQLSH -h 127.0.0.1 -p $YSQL_PORT -U yugabyte -c "ALTER USER ann SET maintenance_work_mem = '4GB';" 2>/dev/null || true
$YSQLSH -h 127.0.0.1 -p $YSQL_PORT -U yugabyte -c "ALTER USER ann SET max_parallel_maintenance_workers = 0;" 2>/dev/null || true

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
