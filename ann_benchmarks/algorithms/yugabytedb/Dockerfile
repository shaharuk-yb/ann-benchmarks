FROM yugabytedb/yugabyte:2025.2.0.0-b131

USER root

# Attempt to install python3 and dependencies using various package managers
RUN (microdnf install -y python3 python3-devel python3-pip git gcc gcc-c++ tar net-tools procps shadow-utils || \
     yum install -y python3 python3-devel python3-pip git gcc gcc-c++ tar net-tools procps shadow-utils || \
     (apt-get update && apt-get install -y python3-pip python3-dev build-essential git net-tools procps))

# Create yugabyte user if it doesn't exist
RUN id -u yugabyte &>/dev/null || useradd -ms /bin/bash yugabyte
RUN chown -R yugabyte:yugabyte /home/yugabyte

WORKDIR /home/app

# Copy root requirements and install
COPY requirements.txt .
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -r requirements.txt

# Install driver
RUN python3 -m pip install psycopg[binary] pgvector

# Copy runner
COPY run_algorithm.py .

# Entrypoint
COPY ann_benchmarks/algorithms/yugabytedb/entrypoint.sh .
RUN chmod u+x entrypoint.sh

# Copy gflag files for master and tserver
COPY ann_benchmarks/algorithms/yugabytedb/masterflagfile .
COPY ann_benchmarks/algorithms/yugabytedb/tserverflagfile .

ENTRYPOINT ["/home/app/entrypoint.sh"]
