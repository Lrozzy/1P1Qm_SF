#!/usr/bin/env python3
"""
Collect and tabulate AUC results from the dimension cutoff vs wires experiments
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import re
import argparse

def extract_auc_from_log(log_file):
    """Extract final test AUC from log file"""
    try:
        with open(log_file, 'r') as f:
            content = f.read()
            # Look for the final test AUC line
            lines = content.split('\n')
            for line in reversed(lines):
                if 'Final test AUC:' in line:
                    auc_match = re.search(r'Final test AUC:\s*([\d.]+)', line)
                    if auc_match:
                        return float(auc_match.group(1))
    except Exception as e:
        print(f"Error reading {log_file}: {e}")
    return None

def is_high_memory_job(dim, wire):
    """Determine if a job configuration uses high memory (CPU) instead of GPU"""
    # High memory needed for: dim>=9 and wires>=5, or dim>=8 and wires>=7, etc.
    # This matches the logic from generate_experiment_jobs.sh
    if ((dim >= 9 and wire >= 5) or 
        (dim >= 8 and wire >= 7) or 
        (dim >= 7 and wire >= 8) or 
        (dim >= 6 and wire >= 9)):
        return True
    return False



def collect_results():
    """Collect results from all experiment runs"""
    # Define ranges
    dim_cutoffs = range(3, 11)
    wires_range = range(3, 11)
    
    # Memory limits
    memory_limits = {10: 5, 9: 6, 8: 7, 7: 8, 6: 9, 5: 10, 4: 10, 3: 10}
    
    # Create results matrix
    results = pd.DataFrame(index=dim_cutoffs, columns=wires_range, dtype=object)
    
    base_log_dir = "/rds/general/user/lr1424/home/1P1Qm_SF/sf_refactor/sf_main_logs/dim_vs_wire_table"
    
    total_expected = 0
    found_results = 0
    
    for dim in dim_cutoffs:
        for wire in wires_range:
            # Check feasibility
            if wire > memory_limits.get(dim, 10):
                results.loc[dim, wire] = "NF"
                continue
            
            total_expected += 1
            job_name = f"dim{dim}_wires{wire}"
            
            # Extract AUC from log file
            log_file = f"{base_log_dir}/{job_name}"
            auc = None
            if os.path.exists(log_file):
                auc = extract_auc_from_log(log_file)
            
            # Store result
            if auc is not None:
                results.loc[dim, wire] = f"{auc:.4f}"
                found_results += 1
            else:
                # Check if job exists but hasn't finished
                log_file = f"{base_log_dir}/{job_name}"
                if os.path.exists(log_file):
                    # Check if job is still running or failed
                    try:
                        with open(log_file, 'r') as f:
                            content = f.read()
                            if "Job finished at" in content:
                                # Job finished but no AUC found = likely out of memory
                                results.loc[dim, wire] = "NF"
                            elif "PBS: job killed: walltime" in content and "exceeded limit" in content:
                                # Job was killed due to walltime limit
                                results.loc[dim, wire] = "TO"
                            else:
                                results.loc[dim, wire] = "RUNNING"
                    except:
                        results.loc[dim, wire] = "ERROR"
                else:
                    results.loc[dim, wire] = "NOT_RUN"
    
    print(f"Found results for {found_results}/{total_expected} expected jobs")
    return results

def print_table(results):
    """Print a nicely formatted table"""
    print("\n" + "="*80)
    print("AUC RESULTS TABLE: Dimension Cutoff vs Number of Wires")
    print("="*80)
    print("Rows: Dimension Cutoff")
    print("Columns: Number of Wires") 
    print("Values: Test AUC (4 decimal places)")
    print("\nHardware Configuration:")
    print("  HIGH-MEMORY CPU jobs (4TB RAM, 8 cores, no GPU, 48h walltime):")
    print("    - dim10_wires5, dim9_wires[5-6], dim8_wires[7], dim7_wires[8], dim6_wires[9]")
    print("  GPU jobs (64GB RAM, 4 cores, 1 GPU, 8h walltime):")
    print("    - All other combinations")
    print("\nLegend:")
    print("  NF      = Not Feasible (exceeds memory limits or crashed due to memory)")
    print("  TO      = Timed Out (exceeded walltime limit)")
    print("  RUNNING = Job currently running")
    print("  NOT_RUN = Job not submitted yet")
    print("  ERROR   = Error reading log file")
    print("-"*80)
    
    # Print header
    print(f"{'Dim':<4}", end="")
    for wire in results.columns:
        print(f"{wire:>9}", end="")
    print()
    
    # Print separator
    print("-"*80)
    
    # Print data rows
    for dim in results.index:
        print(f"{dim:<4}", end="")
        for wire in results.columns:
            value = results.loc[dim, wire]
            if pd.isna(value):
                value = "---"
            
            # Add marker for high-memory jobs that actually ran (have log files)
            if is_high_memory_job(dim, wire):
                base_log_dir = "/rds/general/user/lr1424/home/1P1Qm_SF/sf_refactor/sf_main_logs/dim_vs_wire_table"
                job_name = f"dim{dim}_wires{wire}"
                log_file = f"{base_log_dir}/{job_name}"
                if os.path.exists(log_file) and value not in ["---"]:
                    value = f"{value}*"  # Add asterisk for high-memory jobs that ran
            
            print(f"{value:>9}", end="")
        print()
    
    print("-"*80)
    print("* = High-memory CPU job that ran (4TB RAM, no GPU)")
    
    # Print summary statistics
    numeric_results = []
    for dim in results.index:
        for wire in results.columns:
            val = results.loc[dim, wire]
            if val != "NF" and val not in ["RUNNING", "TO", "NOT_RUN", "ERROR"] and not pd.isna(val):
                try:
                    numeric_results.append(float(val))
                except:
                    pass
    
    if numeric_results:
        print(f"\nSummary Statistics ({len(numeric_results)} completed runs):")
        print(f"  Best AUC:  {max(numeric_results):.4f}")
        print(f"  Worst AUC: {min(numeric_results):.4f}")
        print(f"  Mean AUC:  {np.mean(numeric_results):.4f}")
        print(f"  Std AUC:   {np.std(numeric_results):.4f}")

def save_results(results, filename="auc_results_table.csv", cli_test=False):
    """Save results to CSV file"""
    if not cli_test:
        results.to_csv(filename)
        print(f"\nResults saved to {filename}")
    else:
        print(f"\nSkipping file save (cli_test mode)")

def main():
    parser = argparse.ArgumentParser(description="Collect and tabulate AUC results from experiments")
    parser.add_argument('--cli_test', action='store_true', 
                       help='Run in CLI test mode (no file saving, terminal output only)')
    args = parser.parse_args()
    
    print("Collecting experiment results...")
    results = collect_results()
    print_table(results)
    save_results(results, cli_test=args.cli_test)
    
    if not args.cli_test:
        # Also save a more detailed report
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        detailed_filename = f"experiment_report_{timestamp}.txt"
        
        with open(detailed_filename, 'w') as f:
            f.write("Dimension Cutoff vs Wires Experiment Report\n")
            f.write("="*50 + "\n")
            f.write(f"Generated: {pd.Timestamp.now()}\n\n")
            
        # Count different statuses
        status_counts = {}
        for dim in results.index:
            for wire in results.columns:
                val = results.loc[dim, wire]
                if val == "NF":
                    status = "Not Feasible"
                elif val == "TO":
                    status = "Timed Out"
                elif val in ["RUNNING", "NOT_RUN", "ERROR"]:
                    status = val.title().replace("_", " ")
                elif pd.isna(val):
                    status = "Unknown"
                else:
                    status = "Completed"
                
                status_counts[status] = status_counts.get(status, 0) + 1
        
        f.write("Job Status Summary:\n")
        for status, count in status_counts.items():
            f.write(f"  {status}: {count}\n")
        f.write("\n")
            
        # Write the table
        f.write("Results Table:\n")
        f.write("-" * 50 + "\n")
        # Simple text version of the table
        f.write(results.to_string())
        f.write("\n")
        
        print(f"Detailed report saved to {detailed_filename}")
    else:
        print("Skipping detailed report save (cli_test mode)")

if __name__ == "__main__":
    main()
