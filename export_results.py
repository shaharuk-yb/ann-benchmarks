#!/usr/bin/env python3
"""
Export ANN benchmark results from HDF5 files to CSV format.

This script:
1. Reads HDF5 result files from the specified results directory
2. Calculates metrics (QPS, recall, percentiles, etc.) similar to plot.py
3. Exports both raw data and consolidated metrics to CSV files

Usage:
    python export_results.py --results-dir results/glove-100-angular
    python export_results.py --results-dir results/glove-100-angular --dataset glove-100-angular
"""

import argparse
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Iterator

import h5py
import numpy as np
import pandas as pd


def get_recall_values(dataset_distances: np.ndarray, 
                      run_distances: np.ndarray, 
                      count: int, 
                      epsilon: float = 1e-3) -> Tuple[float, float, np.ndarray]:
    """
    Calculate k-NN recall values.
    
    Args:
        dataset_distances: Ground truth distances from the dataset
        run_distances: Distances returned by the algorithm
        count: Number of neighbors (k)
        epsilon: Small value for threshold comparison
    
    Returns:
        Tuple of (mean_recall, std_recall, recalls_array)
    """
    recalls = np.zeros(len(run_distances))
    for i in range(len(run_distances)):
        # knn_threshold: data[count - 1] + epsilon
        t = dataset_distances[i][count - 1] + epsilon
        actual = 0
        for d in run_distances[i][:count]:
            if d <= t:
                actual += 1
        recalls[i] = actual
    return (np.mean(recalls) / float(count), 
            np.std(recalls) / float(count), 
            recalls)


def calculate_qps(times: np.ndarray, attrs: Dict[str, Any]) -> Optional[float]:
    """
    Calculate Queries Per Second.
    
    Uses best_search_time from attrs if available, otherwise calculates from times.
    
    Args:
        times: Array of query times
        attrs: Attributes dictionary from HDF5 file
    
    Returns:
        QPS value or None if cannot be calculated
    """
    if "best_search_time" in attrs:
        return 1.0 / attrs["best_search_time"]
    elif len(times) > 0:
        # Fallback: use mean time
        mean_time = np.mean(times)
        if mean_time > 0:
            return 1.0 / mean_time
    return None


def calculate_percentiles(times: np.ndarray) -> Dict[str, float]:
    """
    Calculate time percentiles in milliseconds.
    
    Args:
        times: Array of query times in seconds
    
    Returns:
        Dictionary with p50, p95, p99, p999 values in milliseconds
    """
    return {
        'p50_ms': np.percentile(times, 50.0) * 1000.0,
        'p95_ms': np.percentile(times, 95.0) * 1000.0,
        'p99_ms': np.percentile(times, 99.0) * 1000.0,
        'p999_ms': np.percentile(times, 99.9) * 1000.0,
    }


def load_result_files(results_dir: Path, 
                      batch_mode: bool = False) -> Iterator[Tuple[Path, Dict[str, Any], h5py.File]]:
    """
    Load all HDF5 result files from the specified directory.
    
    Args:
        results_dir: Path to the results directory
        batch_mode: If True, only load batch mode results
    
    Yields:
        Tuple of (file_path, properties_dict, hdf5_file_handle)
    """
    for root, _, files in os.walk(results_dir):
        for filename in files:
            if not filename.endswith(".hdf5"):
                continue
            filepath = Path(root) / filename
            try:
                f = h5py.File(filepath, "r")
                properties = dict(f.attrs)
                # Check batch mode if specified
                if "batch_mode" in properties:
                    if batch_mode != properties["batch_mode"]:
                        f.close()
                        continue
                yield filepath, properties, f
            except Exception as e:
                print(f"Warning: Unable to read {filepath}: {e}")
                traceback.print_exc()


def export_raw_data(hdf5_path: Path, 
                    output_dir: Path, 
                    include_recalls: bool = True) -> Optional[Path]:
    """
    Export raw data from a single HDF5 file to CSV format.
    
    Args:
        hdf5_path: Path to the input HDF5 file
        output_dir: Directory to save the output CSV file
        include_recalls: Whether to include recalls column if available
    
    Returns:
        Path to the output CSV file, or None if export failed
    """
    print(f"  Exporting raw data: {hdf5_path.name}")
    
    try:
        with h5py.File(hdf5_path, 'r') as f:
            data_dict = {}
            
            # Extract distances (2D array)
            if 'distances' in f:
                distances = f['distances'][:]
                if len(distances.shape) == 2:
                    for i in range(distances.shape[1]):
                        data_dict[f'distance_{i}'] = distances[:, i]
                else:
                    data_dict['distance'] = distances.flatten()
            
            # Extract neighbors (2D array)
            if 'neighbors' in f:
                neighbors = f['neighbors'][:]
                if len(neighbors.shape) == 2:
                    for i in range(neighbors.shape[1]):
                        data_dict[f'neighbor_{i}'] = neighbors[:, i]
                else:
                    data_dict['neighbor'] = neighbors.flatten()
            
            # Extract times (1D array)
            if 'times' in f:
                data_dict['time'] = f['times'][:]
            
            # Extract recalls from metrics/knn/recalls if available
            if include_recalls:
                if 'metrics' in f and 'knn' in f['metrics'] and 'recalls' in f['metrics']['knn']:
                    data_dict['recall'] = f['metrics']['knn']['recalls'][:]
            
            # Create DataFrame and save
            df = pd.DataFrame(data_dict)
            output_path = output_dir / f"{hdf5_path.stem}.csv"
            df.to_csv(output_path, index=False, float_format='%.17g')
            
            return output_path
            
    except Exception as e:
        print(f"    Error: {e}")
        return None


def calculate_metrics_for_file(filepath: Path,
                               properties: Dict[str, Any],
                               hdf5_file: h5py.File,
                               dataset_distances: Optional[np.ndarray] = None,
                               count: int = 10) -> Dict[str, Any]:
    """
    Calculate all metrics for a single result file.
    
    Args:
        filepath: Path to the HDF5 file
        properties: Properties/attributes from the HDF5 file
        hdf5_file: Open HDF5 file handle
        dataset_distances: Ground truth distances (needed for recall calculation)
        count: Number of neighbors (k)
    
    Returns:
        Dictionary with all calculated metrics
    """
    result = {
        'filepath': str(filepath),
        'filename': filepath.name,
        'algorithm': properties.get('algo', 'unknown'),
        'algorithm_name': properties.get('name', filepath.stem),
        'count': properties.get('count', count),
    }
    
    # Add build time if available
    if 'build_time' in properties:
        result['build_time_s'] = properties['build_time']
    
    # Add index size if available
    if 'index_size' in properties:
        result['index_size_kb'] = properties['index_size']
    
    # Get times and distances
    times = np.array(hdf5_file['times']) if 'times' in hdf5_file else None
    run_distances = np.array(hdf5_file['distances']) if 'distances' in hdf5_file else None
    
    # Calculate QPS
    if times is not None:
        result['qps'] = calculate_qps(times, properties)
        
        # Calculate percentiles
        percentiles = calculate_percentiles(times)
        result.update(percentiles)
        
        result['total_query_time_s'] = np.sum(times)
        result['num_queries'] = len(times)
    
    # Calculate recall if ground truth is provided
    if dataset_distances is not None and run_distances is not None:
        actual_count = min(count, run_distances.shape[1] if len(run_distances.shape) > 1 else count)
        try:
            mean_recall, std_recall, _ = get_recall_values(
                dataset_distances, run_distances, actual_count
            )
            result['recall_mean'] = mean_recall
            result['recall_std'] = std_recall
        except Exception as e:
            print(f"    Warning: Could not calculate recall: {e}")
    elif 'metrics' in hdf5_file:
        # Try to get cached recall from the file
        if 'knn' in hdf5_file['metrics']:
            knn_metrics = hdf5_file['metrics']['knn']
            if 'mean' in knn_metrics.attrs:
                result['recall_mean'] = knn_metrics.attrs['mean']
            if 'std' in knn_metrics.attrs:
                result['recall_std'] = knn_metrics.attrs['std']
    
    return result


def parse_algorithm_params(filename: str) -> Dict[str, Any]:
    """
    Try to parse algorithm parameters from filename.
    
    Attempts to extract common parameters like M, efConstruction, efSearch.
    
    Args:
        filename: Name of the HDF5 file
    
    Returns:
        Dictionary with parsed parameters
    """
    params = {}
    name = filename.replace('.hdf5', '')
    
    # Try pattern: <dataset>_M_<M>_efConstruction_<efConstruction>_<efSearch>
    pattern = r'_M_(\d+)_efConstruction_(\d+)_(\d+)$'
    match = re.search(pattern, name)
    if match:
        params['M'] = int(match.group(1))
        params['ef_construction'] = int(match.group(2))
        params['ef_search'] = int(match.group(3))
        return params
    
    # Try to extract any numeric parameters from underscored format
    parts = name.split('_')
    for i, part in enumerate(parts):
        if part.isdigit() and i > 0:
            # Use previous part as parameter name if it's not a number
            prev_part = parts[i - 1]
            if not prev_part.isdigit():
                params[prev_part] = int(part)
    
    return params


def load_dataset_distances(dataset_name: str) -> Optional[np.ndarray]:
    """
    Load ground truth distances from the dataset file.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'glove-100-angular')
    
    Returns:
        Array of ground truth distances, or None if not found
    """
    dataset_path = Path("data") / f"{dataset_name}.hdf5"
    if not dataset_path.exists():
        return None
    
    try:
        with h5py.File(dataset_path, "r") as f:
            if "distances" in f:
                return np.array(f["distances"])
    except Exception as e:
        print(f"Warning: Could not load dataset distances: {e}")
    
    return None


def export_results(results_dir: str,
                   output_dir: Optional[str] = None,
                   dataset: Optional[str] = None,
                   count: int = 10,
                   export_raw: bool = True,
                   batch_mode: bool = False) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Export all results from HDF5 files to CSV format.
    
    Args:
        results_dir: Path to the results directory containing HDF5 files
        output_dir: Output directory for CSV files (default: results_dir/csv_output)
        dataset: Dataset name for loading ground truth distances
        count: Number of neighbors (k) for recall calculation
        export_raw: Whether to export raw data CSV files
        batch_mode: If True, only process batch mode results
    
    Returns:
        Tuple of (consolidated_csv_path, raw_output_dir_path)
    """
    results_path = Path(results_dir).resolve()
    
    if not results_path.exists():
        print(f"Error: Results directory does not exist: {results_path}")
        return None, None
    
    # Create output directory
    if output_dir:
        output_path = Path(output_dir).resolve()
    else:
        output_path = results_path / 'csv_output'
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectory for raw data exports
    raw_output_path = output_path / 'raw' if export_raw else None
    if raw_output_path:
        raw_output_path.mkdir(exist_ok=True)
    
    # Load dataset ground truth distances if dataset is specified
    dataset_distances = None
    if dataset:
        print(f"Loading ground truth distances for dataset: {dataset}")
        dataset_distances = load_dataset_distances(dataset)
        if dataset_distances is not None:
            print(f"  Loaded {len(dataset_distances)} ground truth distance vectors")
        else:
            print("  Warning: Could not load ground truth distances, recall will use cached values")
    
    # Find all HDF5 files
    hdf5_files = list(results_path.glob("**/*.hdf5"))
    if not hdf5_files:
        print(f"No HDF5 files found in {results_path}")
        return None, None
    
    print(f"\nFound {len(hdf5_files)} HDF5 files to process")
    print(f"Output directory: {output_path}\n")
    
    # Process each file
    all_metrics = []
    
    for filepath in sorted(hdf5_files):
        print(f"Processing: {filepath.name}")
        
        try:
            with h5py.File(filepath, "r") as f:
                properties = dict(f.attrs)
                
                # Check batch mode
                if "batch_mode" in properties:
                    if batch_mode != properties["batch_mode"]:
                        print(f"  Skipping (batch_mode mismatch)")
                        continue
                
                # Calculate metrics
                metrics = calculate_metrics_for_file(
                    filepath, properties, f, dataset_distances, count
                )
                
                # Try to parse additional parameters from filename
                parsed_params = parse_algorithm_params(filepath.name)
                metrics.update(parsed_params)
                
                all_metrics.append(metrics)
                
                # Print summary
                qps = metrics.get('qps')
                recall = metrics.get('recall_mean')
                print(f"  Algorithm: {metrics['algorithm_name']}")
                if qps:
                    print(f"  QPS: {qps:.2f}")
                if recall is not None:
                    print(f"  Recall: {recall:.4f}")
                else:
                    print(f"  Recall: N/A (no ground truth)")
        
        except Exception as e:
            print(f"  Error: {e}")
            traceback.print_exc()
            continue
        
        # Export raw data if requested
        if export_raw and raw_output_path:
            export_raw_data(filepath, raw_output_path, include_recalls=True)
        
        print()
    
    # Create consolidated DataFrame
    if all_metrics:
        df = pd.DataFrame(all_metrics)
        
        # Reorder columns for better readability
        priority_cols = ['algorithm', 'algorithm_name', 'recall_mean', 'qps', 
                        'p50_ms', 'p95_ms', 'p99_ms', 'p999_ms',
                        'build_time_s', 'index_size_kb', 'num_queries']
        other_cols = [c for c in df.columns if c not in priority_cols]
        ordered_cols = [c for c in priority_cols if c in df.columns] + other_cols
        df = df[ordered_cols]
        
        # Sort by recall descending
        if 'recall_mean' in df.columns:
            df = df.sort_values('recall_mean', ascending=False)
        
        # Save consolidated CSV
        consolidated_path = output_path / 'consolidated_metrics.csv'
        df.to_csv(consolidated_path, index=False, float_format='%.6g')
        
        print(f"=" * 60)
        print(f"Export complete!")
        print(f"Consolidated metrics: {consolidated_path}")
        if raw_output_path:
            print(f"Raw data files: {raw_output_path}")
        print(f"\nTotal files processed: {len(all_metrics)}")
        print(f"\nMetrics summary:")
        print(df.to_string(max_rows=20))
        
        return consolidated_path, raw_output_path
    
    return None, None


def main():
    parser = argparse.ArgumentParser(
        description='Export ANN benchmark results from HDF5 files to CSV format.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - export results from a directory
  python export_results.py --results-dir results/glove-100-angular
  
  # With dataset for accurate recall calculation
  python export_results.py --results-dir results/glove-100-angular --dataset glove-100-angular
  
  # Specify custom output directory
  python export_results.py --results-dir results/glove-100-angular -o ./my_output
  
  # Skip raw data export (only generate consolidated metrics)
  python export_results.py --results-dir results/glove-100-angular --no-raw
  
  # Process batch mode results
  python export_results.py --results-dir results/glove-100-angular --batch
"""
    )
    
    parser.add_argument(
        '--results-dir', '-r',
        type=str,
        required=True,
        help='Path to the results directory containing HDF5 files'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output directory for CSV files (default: <results-dir>/csv_output)'
    )
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        default=None,
        help='Dataset name (e.g., glove-100-angular) for loading ground truth distances'
    )
    parser.add_argument(
        '--count', '-k',
        type=int,
        default=10,
        help='Number of neighbors (k) for recall calculation (default: 10)'
    )
    parser.add_argument(
        '--no-raw',
        action='store_true',
        help='Skip exporting raw data CSV files'
    )
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Process batch mode results only'
    )
    
    args = parser.parse_args()
    
    export_results(
        results_dir=args.results_dir,
        output_dir=args.output,
        dataset=args.dataset,
        count=args.count,
        export_raw=not args.no_raw,
        batch_mode=args.batch
    )


if __name__ == "__main__":
    main()

