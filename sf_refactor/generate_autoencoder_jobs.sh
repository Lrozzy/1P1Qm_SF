#!/bin/bash
# Generate PBS job files for autoencoder: dimension cutoff vs wires experiments

# Define the ranges
dim_cutoffs=(3 4 5 6 7 8 9 10)
wires=(3 4 5 6 7 8 9 10)

# Define memory limits (same feasibility map as classifier)
declare -A memory_limits
memory_limits[10]=5
memory_limits[9]=6
memory_limits[8]=7
memory_limits[7]=8
memory_limits[6]=9
memory_limits[5]=10
memory_limits[4]=10
memory_limits[3]=10

# Function to determine if combination needs high memory (CPU)
needs_high_memory() {
    local dim=$1
    local wire=$2
    # Explicit high-memory regions (diagonal band) + previous top-right + special-case rerun
    if [[ \
          ($dim -ge 9 && $wire -ge 5) \
       || ($dim -eq 8 && $wire -ge 6) \
       || ($dim -eq 7 && $wire -ge 6) \
       || ($dim -eq 6 && $wire -ge 6) \
       || ($dim -eq 5 && $wire -ge 7) \
       || ($dim -eq 4 && $wire -ge 8) \
       || ($dim -eq 3 && $wire -ge 10) \
       ]]; then
        return 0  # true - needs high memory (CPU)
    else
        return 1  # false - can use GPU
    fi
}

# Directories
job_dir="/rds/general/user/lr1424/home/1P1Qm_SF/sf_refactor/experiment_jobs_autoencoder"
log_dir_base="/rds/general/user/lr1424/home/1P1Qm_SF/sf_refactor/sf_auto_logs/dim_vs_wire_table"
mkdir -p "$job_dir" "$log_dir_base"

# Clear any existing job files
rm -f ${job_dir}/*.pbs

echo "Generating PBS job files for autoencoder..."
echo "GPU jobs: <= 64GB memory requirements"
echo "CPU jobs: > 64GB memory requirements (high memory nodes)"
echo ""

gpu_count=0
cpu_count=0

for dim in "${dim_cutoffs[@]}"; do
    for wire in "${wires[@]}"; do
        max_wire=${memory_limits[$dim]}
        if [ $wire -gt $max_wire ]; then
            echo "Skipping dim=$dim, wires=$wire (not feasible - exceeds memory limits)"
            continue
        fi

        job_name="dim${dim}_wires${wire}"
        job_file="${job_dir}/${job_name}.pbs"
        log_file="${log_dir_base}/${job_name}"

        if needs_high_memory $dim $wire; then
            echo "Creating CPU job: ${job_name}.pbs (high memory)"
            ((cpu_count++))
            cat > "$job_file" << EOF
#!/bin/bash

#PBS -l select=1:ncpus=8:mem=4000gb
#PBS -l walltime=48:00:00
#PBS -N ${job_name}
#PBS -j oe
#PBS -o ${log_file}

cd \$PBS_O_WORKDIR

# Activate Conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate qml-env

echo "--------------------"
echo "Job started on \$(hostname) at \$(date)"
echo "Job ID: \${PBS_JOBID}"
echo "Configuration: dim_cutoff=${dim}, wires=${wire} (CPU/High-Memory)"
echo "--------------------"

python /rds/general/user/lr1424/home/1P1Qm_SF/sf_refactor/sf_autoencoder.py \
    model.dim_cutoff=${dim} \
    model.wires=${wire} \
    model.photon_modes=${wire} \
    training.epochs=20 \
    training.batch_size=1 \
    data.train_jets=1000 \
    data.val_jets=500 \
    data.test_jets=1000 \
    runtime.run_name="${job_name}"

echo "--------------------"
echo "Job finished at \$(date)"
EOF
        else
            echo "Creating GPU job: ${job_name}.pbs"
            ((gpu_count++))
            cat > "$job_file" << EOF
#!/bin/bash

#PBS -l select=1:ncpus=4:mem=64gb:ngpus=1
#PBS -l walltime=8:00:00
#PBS -N ${job_name}
#PBS -j oe
#PBS -o ${log_file}

cd \$PBS_O_WORKDIR

# Activate Conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate qml-env

echo "--------------------"
echo "Job started on \$(hostname) at \$(date)"
echo "Job ID: \${PBS_JOBID}"
echo "Configuration: dim_cutoff=${dim}, wires=${wire} (GPU)"
echo "--------------------"
nvidia-smi || true
which python
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__); print('GPU:', tf.config.list_physical_devices('GPU'))"

# Overwrite previous log to avoid appending old content
rm -f ${log_file}

python /rds/general/user/lr1424/home/1P1Qm_SF/sf_refactor/sf_autoencoder.py \
    model.dim_cutoff=${dim} \
    model.wires=${wire} \
    model.photon_modes=${wire} \
    training.epochs=20 \
    training.batch_size=1 \
    data.train_jets=1000 \
    data.val_jets=500 \
    data.test_jets=1000 \
    runtime.run_name="${job_name}"

echo "--------------------"
echo "Job finished at \$(date)"
EOF
        fi
    done
done

echo ""
echo "Summary:"
echo "GPU jobs created: $gpu_count"
echo "CPU (high-memory) jobs created: $cpu_count"
echo "Total jobs created: $((gpu_count + cpu_count))"
echo "Job files in: $job_dir"
echo "Logs will write to: $log_dir_base"
