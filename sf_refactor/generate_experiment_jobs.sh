#!/bin/bash
# Generate PBS job files for dimension cutoff vs wires experiments

# Define the ranges
dim_cutoffs=(3 4 5 6 7 8 9 10)
wires=(3 4 5 6 7 8 9 10)

# Define memory limits (increased by 1 from original estimates)
declare -A memory_limits
memory_limits[10]=5  # dim=10, max_wires=5
memory_limits[9]=6   # dim=9, max_wires=6  
memory_limits[8]=7   # dim=8, max_wires=7
memory_limits[7]=8   # dim=7, max_wires=8
memory_limits[6]=9   # dim=6, max_wires=9
memory_limits[5]=10  # dim=5, max_wires=10
memory_limits[4]=10  # dim=4, max_wires=10
memory_limits[3]=10  # dim=3, max_wires=10

# Function to determine if combination needs high memory (CPU)
needs_high_memory() {
    local dim=$1
    local wire=$2
    
    # High memory needed for: dim>=9 and wires>=5, or dim>=8 and wires>=7, etc.
    # Basically anything that would require more than 64GB
    if [[ ($dim -ge 9 && $wire -ge 5) || ($dim -ge 8 && $wire -ge 7) || ($dim -ge 7 && $wire -ge 8) || ($dim -ge 6 && $wire -ge 9) ]]; then
        return 0  # true - needs high memory
    else
        return 1  # false - can use GPU
    fi
}

# Base directory for job files
job_dir="/rds/general/user/lr1424/home/1P1Qm_SF/sf_refactor/experiment_jobs"
mkdir -p $job_dir

# Clear any existing job files
rm -f ${job_dir}/*.pbs

echo "Generating PBS job files..."
echo "GPU jobs: <= 64GB memory requirements"
echo "CPU jobs: > 64GB memory requirements (high memory nodes)"
echo ""

gpu_count=0
cpu_count=0

for dim in "${dim_cutoffs[@]}"; do
    for wire in "${wires[@]}"; do
        # Check if this combination is feasible
        max_wire=${memory_limits[$dim]}
        if [ $wire -gt $max_wire ]; then
            echo "Skipping dim=$dim, wires=$wire (not feasible - exceeds memory limits)"
            continue
        fi
        
        # Determine if this needs high memory (CPU) or can use GPU
        job_name="dim${dim}_wires${wire}"
        job_file="${job_dir}/${job_name}.pbs"
        
        if needs_high_memory $dim $wire; then
            echo "Creating CPU job: ${job_name}.pbs (high memory)"
            ((cpu_count++))
            
            # Generate CPU job file (based on high_memory.pbs)
            cat > $job_file << EOF
#!/bin/bash

#PBS -l select=1:ncpus=8:mem=4000gb
#PBS -l walltime=48:00:00
#PBS -N ${job_name}
#PBS -j oe
#PBS -o /rds/general/user/lr1424/home/1P1Qm_SF/sf_refactor/sf_main_logs/${job_name}

cd \$PBS_O_WORKDIR

# Activate Conda environment
echo "Initializing Conda for this shell..."
source ~/miniconda3/etc/profile.d/conda.sh

echo "Activating Conda environment: qml-env"
conda activate qml-env

# --- Job Diagnostics ---
echo "--------------------"
echo "Job started on \$(hostname) at \$(date)"
echo "Job ID: \${PBS_JOBID}"
echo "Configuration: dim_cutoff=${dim}, wires=${wire} (CPU/High-Memory)"
echo "--------------------"

# --- Run Your Python Script ---
echo "Starting Python script..."
python /rds/general/user/lr1424/home/1P1Qm_SF/sf_refactor/sf_main.py \\
    model.dim_cutoff=${dim} \\
    model.wires=${wire} \\
    model.photon_modes=${wire} \\
    training.epochs=20 \\
    training.batch_size=1 \\
    data.train_jets=1000 \\
    data.val_jets=500 \\
    data.test_jets=1000 \\
    runtime.run_name="${job_name}"

# --- End of Job ---
echo "--------------------"
echo "Job finished at \$(date)"
EOF

        else
            echo "Creating GPU job: ${job_name}.pbs"
            ((gpu_count++))
            
            # Generate GPU job file (based on gpu_job.pbs)
            cat > $job_file << EOF
#!/bin/bash

#PBS -l select=1:ncpus=4:mem=64gb:ngpus=1
#PBS -l walltime=8:00:00
#PBS -N ${job_name}
#PBS -j oe
#PBS -o /rds/general/user/lr1424/home/1P1Qm_SF/sf_refactor/sf_main_logs/${job_name}

cd \$PBS_O_WORKDIR

# Activate Conda environment
echo "Initializing Conda for this shell..."
source ~/miniconda3/etc/profile.d/conda.sh

echo "Activating Conda environment: qml-env"
conda activate qml-env

# --- Job Diagnostics ---
echo "--------------------"
echo "Job started on \$(hostname) at \$(date)"
echo "Job ID: \${PBS_JOBID}"
echo "Configuration: dim_cutoff=${dim}, wires=${wire} (GPU)"
echo "--------------------"
echo "Checking GPU status with nvidia-smi:"
nvidia-smi
echo "--------------------"
echo "Checking Python and package versions:"
which python
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); print('GPU available:', tf.config.list_physical_devices('GPU'))"
echo "--------------------"

# --- Run Your Python Script ---
echo "Starting Python script..."
python /rds/general/user/lr1424/home/1P1Qm_SF/sf_refactor/sf_main.py \\
    model.dim_cutoff=${dim} \\
    model.wires=${wire} \\
    model.photon_modes=${wire} \\
    training.epochs=20 \\
    training.batch_size=1 \\
    data.train_jets=1000 \\
    data.val_jets=500 \\
    data.test_jets=1000 \\
    runtime.run_name="${job_name}"

# --- End of Job ---
echo "--------------------"
echo "Job finished at \$(date)"
EOF

        fi
    done
done

echo ""
echo "Summary:"
echo "========="
echo "GPU jobs created: $gpu_count"
echo "CPU (high-memory) jobs created: $cpu_count"
echo "Total jobs created: $((gpu_count + cpu_count))"
echo ""
echo "All job files generated in: $job_dir"
echo ""
echo "To submit all jobs, run:"
echo "  cd $job_dir"
echo "  for job in *.pbs; do qsub \$job; sleep 2; done"
echo ""
echo "To submit jobs by type:"
echo "  # GPU jobs (faster, shorter walltime):"
echo "  for job in dim3_*.pbs dim4_*.pbs; do qsub \$job; sleep 1; done"
echo "  # CPU jobs (slower, longer walltime):"
echo "  for job in dim*_wires[89].pbs dim*_wires10.pbs; do qsub \$job; sleep 2; done"
