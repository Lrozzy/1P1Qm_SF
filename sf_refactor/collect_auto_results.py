#!/usr/bin/env python3
"""
Collect and tabulate anomaly-score AUC results for the autoencoder experiments
"""

import os
import pandas as pd
import numpy as np
import re

BASE_LOG_DIR = "/rds/general/user/lr1424/home/1P1Qm_SF/sf_refactor/sf_auto_logs/dim_vs_wire_table"

def extract_auc_from_log(log_file):
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

def collect_results():
    dim_cutoffs = range(3, 11)
    wires_range = range(3, 11)
    memory_limits = {10: 5, 9: 6, 8: 7, 7: 8, 6: 9, 5: 10, 4: 10, 3: 10}
    results = pd.DataFrame(index=dim_cutoffs, columns=wires_range, dtype=object)

    for dim in dim_cutoffs:
        for wire in wires_range:
            if wire > memory_limits.get(dim, 10):
                results.loc[dim, wire] = "NF"
                continue
            job_name = f"dim{dim}_wires{wire}"
            log_file = f"{BASE_LOG_DIR}/{job_name}"
            auc = extract_auc_from_log(log_file) if os.path.exists(log_file) else None
            if auc is not None:
                results.loc[dim, wire] = f"{auc:.4f}"
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
    return results

def main():
    results = collect_results()
    print(results)
    results.to_csv("autoencoder_auc_results_table.csv")
    print("Saved to autoencoder_auc_results_table.csv")

if __name__ == "__main__":
    main()
