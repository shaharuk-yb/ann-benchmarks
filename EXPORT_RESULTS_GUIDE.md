# Export Results Guide

This document explains how to use the `export_results.py` script to convert ANN benchmark results from HDF5 format to CSV and calculate performance metrics.

## Overview

The `export_results.py` script provides a unified way to:

1. **Read HDF5 result files** from benchmark runs
2. **Calculate metrics** including:
   - Queries Per Second (QPS)
   - Recall (k-NN accuracy)
   - Latency percentiles (p50, p95, p99, p99.9)
   - Build time and index size
3. **Export data** to CSV format for further analysis

## Requirements

The script uses the following Python packages (already in `requirements.txt`):

- `h5py` - For reading HDF5 files
- `numpy` - For numerical computations
- `pandas` - For data manipulation and CSV export

No additional dependencies are required beyond what's already specified in the project's `requirements.txt`.

## Usage

### Basic Usage

Export results from a directory containing HDF5 files:

```bash
python export_results.py --results-dir results/glove-100-angular
```

This will:
- Process all `.hdf5` files in the specified directory
- Create a `csv_output` folder inside the results directory
- Generate `consolidated_metrics.csv` with all calculated metrics
- Create individual raw data CSV files in `csv_output/raw/`

### With Ground Truth Dataset

For accurate recall calculation, specify the dataset name:

```bash
python export_results.py --results-dir results/glove-100-angular --dataset glove-100-angular
```

The script will load ground truth distances from `data/glove-100-angular.hdf5` and compute recall accurately. Without a dataset, the script uses cached recall values from the HDF5 files (if available).

### Command Line Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--results-dir` | `-r` | Path to directory containing HDF5 result files | *Required* |
| `--output` | `-o` | Output directory for CSV files | `<results-dir>/csv_output` |
| `--dataset` | `-d` | Dataset name for ground truth distances | None |
| `--count` | `-k` | Number of neighbors (k) for recall calculation | 10 |
| `--no-raw` | | Skip exporting individual raw data CSV files | False |
| `--batch` | | Process only batch mode results | False |

### Examples

#### Export with custom output directory

```bash
python export_results.py --results-dir results/sift-128-euclidean -o ./my_analysis/
```

#### Only generate consolidated metrics (skip raw data)

```bash
python export_results.py --results-dir results/glove-100-angular --no-raw
```

#### Process batch mode results

```bash
python export_results.py --results-dir results/glove-100-angular --batch
```

#### Specify different k value

```bash
python export_results.py --results-dir results/glove-100-angular --count 100
```

## Output Files

### Consolidated Metrics (`consolidated_metrics.csv`)

Contains one row per HDF5 file with the following columns:

| Column | Description |
|--------|-------------|
| `algorithm` | Algorithm identifier |
| `algorithm_name` | Full algorithm name with parameters |
| `recall_mean` | Mean k-NN recall (0.0 to 1.0) |
| `qps` | Queries per second |
| `p50_ms` | 50th percentile latency (milliseconds) |
| `p95_ms` | 95th percentile latency (milliseconds) |
| `p99_ms` | 99th percentile latency (milliseconds) |
| `p999_ms` | 99.9th percentile latency (milliseconds) |
| `build_time_s` | Index build time (seconds) |
| `index_size_kb` | Index size (kilobytes) |
| `num_queries` | Number of queries processed |
| `total_query_time_s` | Total query time (seconds) |
| `filepath` | Path to the source HDF5 file |
| `M`, `ef_construction`, `ef_search` | Algorithm-specific parameters (if parseable) |

### Raw Data Files (`raw/*.csv`)

Individual CSV files for each HDF5 file containing:

- `distance_0`, `distance_1`, ... - Distances to each neighbor
- `neighbor_0`, `neighbor_1`, ... - Neighbor indices
- `time` - Query time for each query
- `recall` - Recall value for each query (if available)

## Handling Missing Data

The script gracefully handles missing data:

- **No recall column**: If the HDF5 file doesn't contain pre-computed recalls and no dataset is specified, the recall columns will be missing or show cached values.
- **No ground truth dataset**: A warning is printed, and cached recall values from the HDF5 files are used when available.
- **Corrupted files**: Errors are logged and the script continues processing other files.

## Integration with Existing Tools

This script is designed to complement the existing `plot.py` functionality:

- **plot.py**: Generates visual plots from results
- **export_results.py**: Exports numerical data to CSV for further analysis

Both scripts read from the same HDF5 result files and use similar metric calculations.

## Example Workflow

1. **Run benchmarks** to generate HDF5 result files:
   ```bash
   python run.py --algorithm hnswlib --dataset glove-100-angular
   ```

2. **Export results** to CSV:
   ```bash
   python export_results.py -r results/glove-100-angular -d glove-100-angular
   ```

3. **Analyze** the CSV files in your preferred tool (Excel, pandas, R, etc.)

4. **Generate plots** if needed:
   ```bash
   python plot.py --dataset glove-100-angular
   ```

## Troubleshooting

### "No HDF5 files found"

Ensure the `--results-dir` path is correct and contains `.hdf5` files. The script searches recursively in subdirectories.

### "Could not load ground truth distances"

The dataset file wasn't found at `data/<dataset-name>.hdf5`. Either:
- Download the dataset first
- Run without `--dataset` flag to use cached recall values

### Missing recall values

If recall values are missing:
1. Ensure the dataset is specified with `--dataset`
2. Or, re-run benchmarks with the `--recompute` flag to recalculate metrics

## Performance Notes

- Large result directories may take a few seconds to process
- Raw data export can be skipped with `--no-raw` for faster processing
- The script uses minimal memory by processing files one at a time

