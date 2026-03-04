#!/usr/bin/env python3
"""
GPU Triangle Counting Benchmark Script
Runs adaptive, s=1, and s=7 configurations for multiple graphs
"""

import subprocess
import re
import sys
from statistics import mean, stdev

# Configuration
EXECUTABLE = "./tc_adaptive"
ADJACENCY_MATRIX_LEN = 8192
ITERATIONS = 10

# Dataset configurations
DATASETS = [
    {"name": "roadNet-CA", "file": "roadNet-CA.mtx", "format": "-m"},
    {"name": "Amazon0302", "file": "Amazon0302.mtx", "format": "-m"},
    {"name": "wiki-vote", "file": "wiki-vote.mtx", "format": "-m"},
    {"name": "soc-epinions1", "file": "soc-epinions1.mtx", "format": "-m"},
    {"name": "cit-patents", "file": "cit-Patents.mtx", "format": "-m"},
    {"name": "delaunay-n24", "file": "delaunay_n24.mtx", "format": "-m"},
]

# Test configurations
CONFIGS = [
    {"name": "Adaptive", "flags": ["-A"]},
    {"name": "Fixed s=1", "flags": ["-s", "1"]},
    {"name": "Fixed s=7", "flags": ["-s", "7"]},
]


def run_benchmark(dataset, config):
    """Run a single benchmark and extract execution time"""
    cmd = [
        EXECUTABLE,
        dataset["format"],
        dataset["file"],
        "-a", str(ADJACENCY_MATRIX_LEN),
        "-l", str(ITERATIONS)
    ] + config["flags"]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode != 0:
            print(f"Error running {dataset['name']} with {config['name']}: {result.stderr}")
            return None
        
        # Parse output to extract GPU execution times
        times = []
        # Look for lines with execution data (skip the header and first warmup run)
        lines = result.stdout.strip().split('\n')
        
        for line in lines:
            # Match lines with timing data (contains numeric values)
            if dataset['file'] in line and not line.startswith('graph'):
                # Extract GPU exec time (column 9)
                parts = line.split()
                if len(parts) >= 9:
                    try:
                        gpu_exec_time = float(parts[8])
                        times.append(gpu_exec_time)
                    except (ValueError, IndexError):
                        continue
        
        if not times:
            print(f"Warning: No timing data found for {dataset['name']} with {config['name']}")
            return None
            
        return times
        
    except subprocess.TimeoutExpired:
        print(f"Timeout running {dataset['name']} with {config['name']}")
        return None
    except Exception as e:
        print(f"Exception running {dataset['name']} with {config['name']}: {e}")
        return None


def format_time(seconds):
    """Format time in appropriate units"""
    if seconds < 0.001:
        return f"{seconds * 1000000:.2f} µs"
    elif seconds < 1.0:
        return f"{seconds * 1000:.2f} ms"
    else:
        return f"{seconds:.3f} s"


def main():
    print("="*80)
    print("GPU Triangle Counting Benchmark")
    print("="*80)
    print(f"Executable: {EXECUTABLE}")
    print(f"Adjacency Matrix Length: {ADJACENCY_MATRIX_LEN}")
    print(f"Iterations per configuration: {ITERATIONS}")
    print(f"Datasets: {len(DATASETS)}")
    print(f"Configurations: {len(CONFIGS)}")
    print("="*80)
    print()
    
    results = {}
    
    # Run benchmarks
    for dataset in DATASETS:
        print(f"\n{'='*80}")
        print(f"Dataset: {dataset['name']}")
        print(f"{'='*80}")
        
        dataset_results = {}
        
        for config in CONFIGS:
            print(f"  Running {config['name']}...", end=" ", flush=True)
            
            times = run_benchmark(dataset, config)
            
            if times:
                avg_time = mean(times)
                std_time = stdev(times) if len(times) > 1 else 0
                dataset_results[config['name']] = {
                    'times': times,
                    'avg': avg_time,
                    'std': std_time
                }
                print(f"✓ Avg: {format_time(avg_time)}, Std: {format_time(std_time)}")
            else:
                print("✗ Failed")
                dataset_results[config['name']] = None
        
        results[dataset['name']] = dataset_results
    
    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"{'Dataset':<20} {'Adaptive':<20} {'Fixed s=1':<20} {'Fixed s=7':<20}")
    print("-"*80)
    
    for dataset in DATASETS:
        row = f"{dataset['name']:<20}"
        
        for config in CONFIGS:
            result = results[dataset['name']].get(config['name'])
            if result:
                row += f" {format_time(result['avg']):<20}"
            else:
                row += f" {'N/A':<20}"
        
        print(row)
    
    # Print speedup analysis
    print("\n" + "="*80)
    print("SPEEDUP ANALYSIS (Relative to Adaptive)")
    print("="*80)
    print(f"{'Dataset':<20} {'s=1 vs Adaptive':<25} {'s=7 vs Adaptive':<25}")
    print("-"*80)
    
    for dataset in DATASETS:
        adaptive_result = results[dataset['name']].get('Adaptive')
        s1_result = results[dataset['name']].get('Fixed s=1')
        s7_result = results[dataset['name']].get('Fixed s=7')
        
        row = f"{dataset['name']:<20}"
        
        if adaptive_result and s1_result:
            speedup = s1_result['avg'] / adaptive_result['avg']
            row += f" {speedup:.2f}× slower{'':<13}"
        else:
            row += f" {'N/A':<25}"
        
        if adaptive_result and s7_result:
            speedup = s7_result['avg'] / adaptive_result['avg']
            if speedup > 1.0:
                row += f" {speedup:.2f}× slower"
            else:
                row += f" {1/speedup:.2f}× faster"
        else:
            row += f" {'N/A':<25}"
        
        print(row)
    
    # Export results to CSV
    csv_filename = "benchmark_results.csv"
    print(f"\n{'='*80}")
    print(f"Exporting results to {csv_filename}")
    print("="*80)
    
    with open(csv_filename, 'w') as f:
        # Header
        f.write("Dataset,Adaptive_Avg(s),Adaptive_Std(s),s1_Avg(s),s1_Std(s),s7_Avg(s),s7_Std(s)\n")
        
        # Data rows
        for dataset in DATASETS:
            adaptive = results[dataset['name']].get('Adaptive')
            s1 = results[dataset['name']].get('Fixed s=1')
            s7 = results[dataset['name']].get('Fixed s=7')
            
            f.write(f"{dataset['name']},")
            f.write(f"{adaptive['avg']:.6f},{adaptive['std']:.6f}," if adaptive else "N/A,N/A,")
            f.write(f"{s1['avg']:.6f},{s1['std']:.6f}," if s1 else "N/A,N/A,")
            f.write(f"{s7['avg']:.6f},{s7['std']:.6f}\n" if s7 else "N/A,N/A\n")
    
    print(f"✓ Results exported to {csv_filename}")
    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()