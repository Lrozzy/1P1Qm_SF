#!/usr/bin/env python3
"""
Collect and tabulate anomaly-score AUC results for the autoencoder experiments
"""

import os
import pandas as pd
import numpy as np
import re
import argparse

BASE_LOG_DIR = "/rds/general/user/lr1424/home/1P1Qm_SF/sf_refactor/sf_auto_logs/dim_vs_wire_table"

def extract_auc_from_log(log_file):
    """Extract final test AUC from autoencoder log file"""
    try:
        with open(log_file, 'r') as f:
            content = f.read()
            for line in reversed(content.splitlines()):
                if 'Final test AUC (anomaly score):' in line:
                    m = re.search(r'Final test AUC \(anomaly score\):\s*([\d.]+)', line)
                    if m:
                        return float(m.group(1))
                if 'Final test AUC:' in line:  # fallback pattern
                    m = re.search(r'Final test AUC:\s*([\d.]+)', line)
                    if m:
                        return float(m.group(1))
    except Exception:
        pass
    return None

def is_high_memory_job(dim, wire):
    """Match the high-memory rule used in generate_autoencoder_jobs.sh"""
    if ((dim >= 9 and wire >= 5) or
        (dim == 8 and wire >= 6) or
        (dim == 7 and wire >= 6) or
        (dim == 6 and wire >= 6) or
        (dim == 5 and wire >= 7) or
        (dim == 4 and wire >= 8) or
        (dim == 3 and wire >= 10) or
        (dim == 3 and wire == 9)):
        return True
    return False

def collect_results():
    """Collect results from all autoencoder runs"""
    dim_cutoffs = range(3, 11)
    wires_range = range(3, 11)
    memory_limits = {10: 5, 9: 6, 8: 7, 7: 8, 6: 9, 5: 10, 4: 10, 3: 10}

    results = pd.DataFrame(index=dim_cutoffs, columns=wires_range, dtype=object)

    total_expected = 0
    found_results = 0

    for dim in dim_cutoffs:
        for wire in wires_range:
            # feasibility check
            if wire > memory_limits.get(dim, 10):
                results.loc[dim, wire] = "NF"
                continue

            total_expected += 1
            job_name = f"dim{dim}_wires{wire}"
            log_file = f"{BASE_LOG_DIR}/{job_name}"

            auc = extract_auc_from_log(log_file) if os.path.exists(log_file) else None

            if auc is not None:
                results.loc[dim, wire] = f"{auc:.4f}"
                found_results += 1
            else:
                if os.path.exists(log_file):
                    try:
                        with open(log_file, 'r') as f:
                            content = f.read()
                            if "Job finished at" in content:
                                results.loc[dim, wire] = "NF"
                            elif "PBS: job killed: walltime" in content and "exceeded limit" in content:
                                results.loc[dim, wire] = "TO"
                            else:
                                results.loc[dim, wire] = "RUNNING"
                    except Exception:
                        results.loc[dim, wire] = "ERROR"
                else:
                    results.loc[dim, wire] = "NOT_RUN"

    print(f"Found results for {found_results}/{total_expected} expected jobs")
    return results

def print_table(results):
    """Pretty-print a table like the classifier collector, with star markers"""
    print("\n" + "="*80)
    print("AUC RESULTS TABLE: Dimension Cutoff vs Number of Wires")
    print("="*80)
    print("Rows: Dimension Cutoff")
    print("Columns: Number of Wires")
    print("Values: Test AUC (anomaly score, 4 decimal places)")
    print("\\nHardware Configuration:")
    print("  HIGH-MEMORY CPU jobs (4TB RAM, 8 cores, no GPU, 48h walltime):")
    print("    - Expanded diagonal: dim8_wires6+, dim7_wires6+, dim6_wires6+, dim5_wires7+, dim4_wires8+, dim3_wires10; plus dim3_wires9 rerun")
    print("  GPU jobs (64GB RAM, 4 cores, 1 GPU, 8h walltime):")
    print("    - All other combinations")
    print("\nLegend:")
    print("  NF      = Not Feasible (exceeds memory limits or crashed due to memory)")
    print("  TO      = Timed Out (exceeded walltime limit)")
    print("  RUNNING = Job currently running")
    print("  NOT_RUN = Job not submitted yet")
    print("  ERROR   = Error reading log file")
    print("-"*87)

    # header
    print(f"{'':>4}{'Wires':>9}", end="")
    for wire in results.columns:
        print(f"{wire:>9}", end="")
    print()
    print(f"{'Dim':<4}{'':>9}", end="")
    for _ in results.columns:
        print(f"{'':>9}", end="")
    print()
    print("-"*87)

    for dim in results.index:
        print(f"{dim:<4}{'':>9}", end="")
        for wire in results.columns:
            value = results.loc[dim, wire]
            if pd.isna(value):
                value = "---"

            # mark high-memory jobs that ran (log file exists)
            if is_high_memory_job(dim, wire):
                job_name = f"dim{dim}_wires{wire}"
                log_file = f"{BASE_LOG_DIR}/{job_name}"
                if os.path.exists(log_file) and value not in ["---"]:
                    value = f"{value}*"

            print(f"{value:>9}", end="")
        print()

    print("-"*87)
    print("* = High-memory CPU job that ran (4TB RAM, no GPU)")

    # summary stats
    numeric = []
    best = None
    worst = None
    for dim in results.index:
        for wire in results.columns:
            val = results.loc[dim, wire]
            if val in ["NF", "RUNNING", "TO", "NOT_RUN", "ERROR"] or pd.isna(val):
                continue
            try:
                auc = float(str(val).replace('*',''))
            except Exception:
                continue
            numeric.append(auc)
            if best is None or auc > best[0]:
                best = (auc, dim, wire)
            if worst is None or auc < worst[0]:
                worst = (auc, dim, wire)

    if numeric:
        print(f"\nSummary Statistics ({len(numeric)} completed runs):")
        if best is not None:
            print(f"  Best AUC:  {best[0]:.4f} (dim: {best[1]}, wires: {best[2]})")
        if worst is not None:
            print(f"  Worst AUC: {worst[0]:.4f} (dim: {worst[1]}, wires: {worst[2]})")
        print(f"  Mean AUC:  {np.mean(numeric):.4f}")
        print(f"  Std AUC:   {np.std(numeric):.4f}")

def save_results(results, filename="autoencoder_auc_results_table.csv", cli_test=False):
    if not cli_test:
        results.to_csv(filename)
        print(f"\nResults saved to {filename}")
    else:
        print("\nSkipping file save (cli_test mode)")

def main():
    parser = argparse.ArgumentParser(description="Collect and tabulate AUC results from autoencoder experiments")
    parser.add_argument('--cli_test', action='store_true', help='Run in CLI test mode (no file saving, terminal output only)')
    args = parser.parse_args()

    print("Collecting experiment results...")
    results = collect_results()
    print_table(results)
    save_results(results, cli_test=args.cli_test)

    if not args.cli_test:
        # Optional detailed report mirroring classifier tool
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        detailed_filename = f"autoencoder_experiment_report_{timestamp}.txt"
        with open(detailed_filename, 'w') as f:
            f.write("Autoencoder Dimension Cutoff vs Wires Experiment Report\n")
            f.write("="*50 + "\n")
            f.write(f"Generated: {pd.Timestamp.now()}\n\n")
            f.write(results.to_string())
            f.write("\n")
        print(f"Detailed report saved to {detailed_filename}")
    else:
        print("Skipping detailed report save (cli_test mode)")

if __name__ == "__main__":
    main()
