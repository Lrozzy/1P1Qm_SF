#!/bin/bash
# Submit all generated autoencoder experiment jobs

job_dir="/rds/general/user/lr1424/home/1P1Qm_SF/sf_refactor/experiment_jobs_autoencoder"

if [ ! -d "$job_dir" ]; then
    echo "Job directory $job_dir does not exist!"
    echo "Run generate_autoencoder_jobs.sh first."
    exit 1
fi

cd "$job_dir"

job_files=(*.pbs)
total_jobs=${#job_files[@]}

if [ $total_jobs -eq 0 ]; then
    echo "No .pbs files found in $job_dir"
    exit 1
fi

echo "Found $total_jobs autoencoder job files to submit"
echo ""

read -p "Submit all jobs at once? (y/n): " submit_all

if [[ $submit_all =~ ^[Yy]$ ]]; then
    echo "Submitting all jobs with 2-second delays..."
    submitted=0
    for job_file in "${job_files[@]}"; do
        if [ -f "$job_file" ]; then
            echo "Submitting: $job_file"
            job_id=$(qsub "$job_file")
            if [ $? -eq 0 ]; then
                echo "  Job ID: $job_id"
                ((submitted++))
            else
                echo "  ERROR: Failed to submit $job_file"
            fi
            sleep 2
        fi
    done
    echo "Submitted $submitted out of $total_jobs jobs"
else
    echo "Options:"
    echo "1. Submit GPU jobs only"
    echo "2. Submit CPU jobs only"
    echo "3. Submit specific job"
    echo "4. Exit"
    read -p "Choose option (1-4): " option
    case $option in
        1)
            echo "Submitting GPU jobs..."
            submitted=0
            for job_file in *.pbs; do
                if grep -q "ngpus=1" "$job_file"; then
                    echo "Submitting GPU job: $job_file"
                    job_id=$(qsub "$job_file")
                    if [ $? -eq 0 ]; then
                        echo "  Job ID: $job_id"
                        ((submitted++))
                    fi
                    sleep 1
                fi
            done
            echo "Submitted $submitted GPU jobs"
            ;;
        2)
            echo "Submitting CPU jobs..."
            submitted=0
            for job_file in *.pbs; do
                if grep -q "mem=4000gb" "$job_file"; then
                    echo "Submitting CPU job: $job_file"
                    job_id=$(qsub "$job_file")
                    if [ $? -eq 0 ]; then
                        echo "  Job ID: $job_id"
                        ((submitted++))
                    fi
                    sleep 2
                fi
            done
            echo "Submitted $submitted CPU jobs"
            ;;
        3)
            echo "Available job files:"
            ls -1 *.pbs
            read -p "Enter job filename: " specific_job
            if [ -f "$specific_job" ]; then
                job_id=$(qsub "$specific_job")
                if [ $? -eq 0 ]; then
                    echo "Submitted: $specific_job with Job ID: $job_id"
                else
                    echo "Failed to submit: $specific_job"
                fi
            else
                echo "File not found: $specific_job"
            fi
            ;;
        4)
            echo "Exiting without submitting jobs"
            exit 0
            ;;
        *)
            echo "Invalid option"
            exit 1
            ;;
    esac
fi

echo ""
echo "Check job status with: qstat -u \$USER"
echo "Monitor logs in: /rds/general/user/lr1424/home/1P1Qm_SF/sf_refactor/sf_auto_logs/dim_vs_wire_table"
